from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import IterableDataset, DataLoader

import tartanair as ta


class TartanAirSceneFlowDataset(IterableDataset):
    """
    TartanAir scene-flow dataset with a RAFT-3D-compatible representation.

    Each sample contains:

        image1      [3, H, W]       float32, 0..255
        image2      [3, H, W]       float32, 0..255

        depth1      [H, W]          float32
        depth2      [H, W]          float32

        flow_gt     [2, H, W]       float32
                    Camera-induced optical flow [du, dv].

        valid_gt    [H, W]          bool
                    Geometrically valid projection mask.

        flowxyz     [H, W, 3]       float32
                    [:, :, 0] = du
                    [:, :, 1] = dv
                    [:, :, 2] = 1/Z2 - 1/Z1

        valid_scene [H, W]          bool

        pose1       [4, 4]          float32
        pose2       [4, 4]          float32

                    Camera-to-world transformations in OpenCV
                    camera coordinates:

                        x = right
                        y = down
                        z = forward

        intrinsics  [4]             float32
                    [fx, fy, cx, cy]

    The original TartanAir poses are provided in the NED convention:

        x = forward
        y = right
        z = down

    pose_to_matrix() re-expresses both the world and camera coordinates
    in the OpenCV convention.
    """

    def __init__(
        self,
        root: str,
        envs: Sequence[str],
        difficulties: Sequence[str] = ("easy",),
        trajectories: Sequence[str] = ("P000",),
        camera: str = "lcam_front",
        frame_sep: int = 1,
        image_size: Sequence[int] | None = (640, 640),
        subset_framenum: int = 200,
        loader_workers: int = 0,
    ):
        super().__init__()

        self.root = root
        self.envs = list(envs)
        self.difficulties = list(difficulties)
        self.trajectories = list(trajectories)
        self.camera = camera
        self.frame_sep = int(frame_sep)
        self.image_size = image_size
        self.subset_framenum = subset_framenum
        self.loader_workers = loader_workers

        if not self.trajectories:
            raise ValueError(
                "trajectories must contain at least one sequence."
            )

        if self.frame_sep < 1:
            raise ValueError(
                "frame_sep must be >= 1."
            )

        ta.init(str(self.root))

    # ------------------------------------------------------------------
    # Pose conversion
    # ------------------------------------------------------------------

    @staticmethod
    def pose_to_matrix(pose):
        """
        Convert a TartanAir pose to a 4x4 camera-to-world matrix in
        OpenCV camera coordinates.

        Input:
            [..., 7]

        [tx, ty, tz, qx, qy, qz, qw]

        Input convention:
            x = forward
            y = right
            z = down

        Output convention:
            x = right
            y = down
            z = forward

        The transformation remains camera-to-world:

            p_world = T_world_cam @ p_cam
        """

        # NED -> OpenCV optical camera coordinates:
        #
        #   NED x -> CV z
        #   NED y -> CV x
        #   NED z -> CV y
        #
        # Therefore:
        #
        #   [x_cv]   [0 1 0] [x_ned]
        #   [y_cv] = [0 0 1] [y_ned]
        #   [z_cv]   [1 0 0] [z_ned]

        ned_to_cv = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        pose = np.asarray(
            pose,
            dtype=np.float64,
        )

        if pose.shape[-1] != 7:
            raise ValueError(
                f"Expected pose with last dimension 7, "
                f"got shape {pose.shape}."
            )

        batch_shape = pose.shape[:-1]

        flat_pose = pose.reshape(-1, 7)

        transforms = np.zeros(
            (flat_pose.shape[0], 4, 4),
            dtype=np.float64,
        )

        transforms[:, :3, :3] = (
            Rotation.from_quat(
                flat_pose[:, 3:7]
            ).as_matrix()
        )

        transforms[:, :3, 3] = flat_pose[:, :3]

        transforms[:, 3, 3] = 1.0

        # Re-express both world and camera coordinates.
        transforms = (
            ned_to_cv
            @ transforms
            @ ned_to_cv.T
        )

        return transforms.reshape(
            batch_shape + (4, 4)
        )

    # ------------------------------------------------------------------
    # TartanAir loader
    # ------------------------------------------------------------------

    def _make_tartanair_loader(self):
        image_shape = (
            None
            if self.image_size is None
            else list(self.image_size)
        )

        return ta.dataloader(
            env=self.envs,
            difficulty=self.difficulties,
            trajectory_id=self.trajectories,

            modality=[
                "image",
                "depth",
                "pose",
            ],

            camera_name=[
                self.camera,
            ],

            new_image_shape_hw=image_shape,

            seq_length={
                "image": 2,
                "depth": 2,
                "pose": 2,
            },

            subset_framenum=self.subset_framenum,

            seq_stride=1,
            frame_skip=self.frame_sep - 1,
            batch_size=1,

            num_workers=self.loader_workers,

            shuffle=False,

            verbose=False,
        )

    # ------------------------------------------------------------------
    # Intrinsics
    # ------------------------------------------------------------------

    def _get_intrinsics(
        self,
        image1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return [fx, fy, cx, cy].

        TartanAir's original 640x640 camera intrinsics are:

            fx = 320
            fy = 320
            cx = 320
            cy = 320

        For pure resizing, all four quantities scale accordingly.
        """

        h, w = image1.shape[-2:]

        fx = 320.0 * w / 640.0
        fy = 320.0 * h / 640.0
        cx = 320.0 * w / 640.0
        cy = 320.0 * h / 640.0

        return torch.tensor(
            [fx, fy, cx, cy],
            dtype=torch.float32,
            device=image1.device,
        )

    # ------------------------------------------------------------------
    # Camera-induced scene flow
    # ------------------------------------------------------------------

    def _compute_scene_flow(
        self,
        depth1: torch.Tensor,
        pose1: torch.Tensor,
        pose2: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """
        Compute camera-induced scene flow analytically from:

            depth1
            pose1
            pose2
            camera intrinsics

        The output uses the RAFT-3D representation:

            flowxyz[..., 0] = du
            flowxyz[..., 1] = dv
            flowxyz[..., 2] = 1/Z2 - 1/Z1

        Camera convention:

            x = right
            y = down
            z = forward

        Poses:

            camera-to-world

        Returns
        -------
        flow:
            [2, H, W]

        valid:
            [H, W]

        flowxyz:
            [H, W, 3]
        """

        depth1 = depth1.float()
        pose1 = pose1.float()
        pose2 = pose2.float()
        intrinsics = intrinsics.float()

        fx, fy, cx, cy = intrinsics

        h, w = depth1.shape

        # --------------------------------------------------------------
        # Pixel coordinates.
        # --------------------------------------------------------------

        v, u = torch.meshgrid(
            torch.arange(
                h,
                device=depth1.device,
                dtype=depth1.dtype,
            ),
            torch.arange(
                w,
                device=depth1.device,
                dtype=depth1.dtype,
            ),
            indexing="ij",
        )

        # --------------------------------------------------------------
        # Valid source depth.
        # --------------------------------------------------------------

        valid_depth = (
            torch.isfinite(depth1)
            & (depth1 > 0)
        )

        # --------------------------------------------------------------
        # Back-project image 1 into 3-D.
        #
        # OpenCV convention:
        #
        #   x = right
        #   y = down
        #   z = forward
        #
        #   x = (u-cx) * z / fx
        #   y = (v-cy) * z / fy
        #   z = depth
        # --------------------------------------------------------------

        z1 = depth1
        x1 = (u - cx) * z1 / fx
        y1 = (v - cy) * z1 / fy

        points1 = torch.stack(
            [
                x1,
                y1,
                z1,
                torch.ones_like(z1),
            ],
            dim=-1,
        )

        # --------------------------------------------------------------
        # Camera 1 -> camera 2.
        #
        # pose1 and pose2 are:
        #
        #   T_world_cam1
        #   T_world_cam2
        #
        # Therefore:
        #
        #   T_cam2_cam1 =
        #       inv(T_world_cam2) @ T_world_cam1
        # --------------------------------------------------------------

        T_cam2_from_cam1 = (
            torch.linalg.inv(pose2)
            @ pose1
        )

        points2 = (
            points1.reshape(-1, 4)
            @ T_cam2_from_cam1.T
        ).reshape(
            h,
            w,
            4,
        )

        x2 = points2[..., 0]
        y2 = points2[..., 1]
        z2 = points2[..., 2]

        # --------------------------------------------------------------
        # Valid transformed points.
        # --------------------------------------------------------------

        valid = (
            valid_depth
            & torch.isfinite(x2)
            & torch.isfinite(y2)
            & torch.isfinite(z2)
            & (z2 > 0)
        )

        # --------------------------------------------------------------
        # Project into image 2.
        # --------------------------------------------------------------

        u2 = fx * x2 / z2 + cx
        v2 = fy * y2 / z2 + cy

        valid &= (
            torch.isfinite(u2)
            & torch.isfinite(v2)
            & (u2 >= 0)
            & (u2 <= w - 1)
            & (v2 >= 0)
            & (v2 <= h - 1)
        )

        # --------------------------------------------------------------
        # Optical flow.
        # --------------------------------------------------------------

        flow_u = u2 - u
        flow_v = v2 - v

        flow = torch.stack(
            [
                flow_u,
                flow_v,
            ],
            dim=0,
        )

        # --------------------------------------------------------------
        # RAFT-3D third component:
        #
        #     1/Z2 - 1/Z1
        #
        # This is inverse-depth change, not metric Z displacement.
        # --------------------------------------------------------------

        flow_z = 1.0 / z2 - 1.0 / z1

        flowxyz = torch.stack(
            [
                flow_u,
                flow_v,
                flow_z,
            ],
            dim=-1,
        )

        # Remove invalid values.
        flow[:, ~valid] = 0.0
        flowxyz[~valid] = 0.0

        return flow, valid, flowxyz

    # ------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------

    def _iterate(
        self,
        tartanair_loader,
    ):
        """
        Iterate over all trajectories supplied to the TartanAir loader.

        TartanAir itself handles the trajectory switching. The returned
        pose modality is used directly for scene-flow computation.
        """

        try:
            while True:

                # ------------------------------------------------------
                # Load next pair.
                # ------------------------------------------------------

                try:
                    batch = tartanair_loader.load_sample()
                except StopIteration:
                    break

                images = batch[
                    f"image_{self.camera}"
                ]

                depths = batch[
                    f"depth_{self.camera}"
                ]

                poses = batch[
                    f"pose_{self.camera}"
                ]

                # ------------------------------------------------------
                # Remove TartanAir batch dimension.
                #
                # Expected:
                #
                #   images: [1, 2, 3, H, W]
                #   depths: [1, 2, H, W]
                #   poses:  [1, 2, 7]
                # ------------------------------------------------------

                images = images[0]
                depths = depths[0]
                poses = poses[0]

                # invert colour channel order BGR -> RGB
                images = images.flip(dims=[1])

                if images.shape[0] != 2:
                    raise ValueError(
                        f"Expected two images, got "
                        f"shape {images.shape}"
                    )

                if depths.shape[0] != 2:
                    raise ValueError(
                        f"Expected two depth maps, got "
                        f"shape {depths.shape}"
                    )

                if poses.shape[0] != 2:
                    raise ValueError(
                        f"Expected two poses, got "
                        f"shape {poses.shape}"
                    )

                image1 = images[0].float()
                image2 = images[1].float()

                depth1 = depths[0].float()
                depth2 = depths[1].float()

                # ------------------------------------------------------
                # Convert poses:
                #
                #   TartanAir NED
                #
                # to:
                #
                #   OpenCV
                #
                # and return camera-to-world transforms.
                # ------------------------------------------------------

                pose1 = torch.from_numpy(
                    self.pose_to_matrix(
                        poses[0].cpu().numpy()
                    )
                ).float()

                pose2 = torch.from_numpy(
                    self.pose_to_matrix(
                        poses[1].cpu().numpy()
                    )
                ).float()

                # ------------------------------------------------------
                # Intrinsics.
                # ------------------------------------------------------

                intrinsics = self._get_intrinsics(
                    image1
                )

                # ------------------------------------------------------
                # Compute camera-induced optical flow and
                # RAFT-3D scene flow analytically.
                # ------------------------------------------------------

                flow_gt, valid_gt, flowxyz = (
                    self._compute_scene_flow(
                        depth1=depth1,
                        pose1=pose1,
                        pose2=pose2,
                        intrinsics=intrinsics,
                    )
                )

                # The scene-flow validity is the same geometric
                # validity mask used for the optical flow.
                valid_scene = valid_gt.clone()

                # ------------------------------------------------------
                # Return sample.
                # ------------------------------------------------------

                yield {
                    "image1": image1,
                    "image2": image2,

                    "depth1": depth1,
                    "depth2": depth2,

                    # Camera-induced optical flow.
                    "flow_gt": flow_gt,
                    "valid_gt": valid_gt,

                    # RAFT-3D scene-flow representation.
                    "flowxyz": flowxyz,
                    "valid_scene": valid_scene,

                    # Camera-to-world poses in OpenCV convention.
                    "pose1": pose1,
                    "pose2": pose2,

                    # [fx, fy, cx, cy]
                    "intrinsics": intrinsics,
                    "frame_sep": self.frame_sep,
                    "camera": self.camera,
                }

        finally:
            tartanair_loader.stop_cachers()

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------

    def __iter__(self):
        tartanair_loader = self._make_tartanair_loader()

        yield from self._iterate(
            tartanair_loader
        )

    def __len__(self):
        """
        Return the number of frame pairs based on the actual number
        of frames found on disk.

        For N frames and separation s:

            number of pairs = N - s
        """

        from os.path import join
        from tartanair.data_cacher.datafile_editor import enumerate_frames

        ta_loader = ta.TartanAirDataLoader(
            self.root
        )

        folderdict = (
            ta_loader.compile_modality_and_cameraname(
                self.difficulties,
                ["image"],
                [self.camera],
            )
        )

        folderlist = list(
            dict.fromkeys(
                fl
                for folders in folderdict.values()
                for fl in folders
            )
        )

        onemodfolder = next(
            (
                fl
                for fl in folderlist
                if not fl.endswith("imu")
            ),
            None,
        )

        if onemodfolder is None:
            raise RuntimeError(
                "Could not determine a frame-based "
                "TartanAir modality folder."
            )

        total = 0

        for env in self.envs:
            for difficulty in self.difficulties:
                for trajectory in self.trajectories:

                    traj_str = join(
                        env,
                        f"Data_{difficulty}",
                        trajectory,
                    )

                    frames = enumerate_frames(
                        join(
                            self.root,
                            traj_str,
                            onemodfolder,
                        )
                    )

                    total += max(
                        0,
                        len(frames) - self.frame_sep,
                    )

        return total

# ======================================================================
# Convenience function
# ======================================================================

def make_dataloader(
    root,
    envs,
    difficulties=("easy",),
    trajectories=("P000",),
    camera="lcam_front",
    frame_sep=1,
    image_size=(640, 640),
    subset_framenum=200,
    loader_workers=0,
    batch_size=1,
):
    dataset = TartanAirSceneFlowDataset(
        root=root,
        envs=envs,
        difficulties=difficulties,
        trajectories=trajectories,
        camera=camera,
        frame_sep=frame_sep,
        image_size=image_size,
        subset_framenum=subset_framenum,
        loader_workers=loader_workers,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )
