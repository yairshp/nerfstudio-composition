# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal, Optional, Union

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_beta"] = "viewer"
    """Type of viewer"""
    checkpoint_path: Optional[str] = None
    """Path to the checkpoint file"""
    #! FG Arguments
    load_fg_config: Optional[Path] = None
    """Path to config YAML file for the fg object"""
    fg_camera_path_filename: Optional[Path] = None
    """Filename of the camera path to render of the fg object."""
    fg_checkpoint_path: Optional[str] = None
    """"Path to checkpoint of the fg."""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
            checkpoint_path=self.checkpoint_path,
        )
        if self.load_fg_config:
            fg_config, fg_pipeline, _, _ = eval_setup(
                self.load_fg_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
                checkpoint_path=self.fg_checkpoint_path,
            )
        if self.fg_camera_path_filename:
            with open(str(self.fg_camera_path_filename), "r", encoding="utf-8") as f:
                fg_camera_data = json.load(f)
            fg_crop_data = get_crop_from_json(fg_camera_data)

        else:
            fg_pipeline = None
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        base_dir = None if self.checkpoint_path is None else self.checkpoint_path[:self.checkpoint_path.find("nerfstudio_models") - 1]

        _start_viewer(config, pipeline, step, base_dir_str=base_dir)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int, base_dir_str: Optional[str] = None, fg_pipeline: Optional[Pipeline] = None, fg_crop_data: Optional[tuple] = None):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    if base_dir_str is None:
        base_dir = config.get_base_dir()
    else:
        if base_dir_str[0] == '/':
            base_dir_components = base_dir_str.split('/')[1:]
            base_dir_components[0] = f"/{base_dir_components[0]}"
        else:
            base_dir_components = base_dir_str.split('/')
        base_dir = Path(*base_dir_components)
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    if config.vis == "viewer":
        viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            fg_pipeline=fg_pipeline,
            fg_crop_data=fg_crop_data,
        )
        banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]
    if config.vis == "viewer_beta":
        viewer_state = ViewerBetaState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=base_dir,
            pipeline=pipeline,
        )
        banner_messages = [f"Viewer Beta at: {viewer_state.viewer_url}"]

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    if isinstance(viewer_state, ViewerState):
        viewer_state.viser_server.set_training_state("completed")
    viewer_state.update_scene(step=step)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewer).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewer)  # noqa
