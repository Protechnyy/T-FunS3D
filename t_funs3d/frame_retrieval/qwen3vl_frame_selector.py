import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """You are an AI system that selects RGB frames for a robot interaction task.

You are given:
1. a robot task,
2. the target object affected by the task,
3. the functional component that the robot must directly manipulate,
4. a sequence of RGB frames.

Select the single frame that is most likely to contain the given functional component and provides the best visual evidence for locating it.

Use the provided functional_component exactly.

You must select exactly one frame from the provided frames, even when the component is small, partially visible, occluded, or difficult to recognize.

Prefer a frame where the functional component itself is visible. Do not select a frame only because it clearly shows the target object.

Output valid JSON only."""

# user prompt 在图片序列前后分成两段，图片按 frame_i 标签插入两段之间
USER_PROMPT_HEADER = """Task: {prompt}
Target object: {target_object}
Functional component: {functional_component}

The following RGB images are labeled frame_0, frame_1, ..., frame_{last_index}."""

USER_PROMPT_QUERY = """Select exactly one frame that is most likely to contain the functional component relevant to this task.

You must choose one of the provided frame labels.

Return valid JSON only:

{
  "frame": "frame_0"
}"""


class Qwen3VLFrameSelector:
    def __init__(self, cfg: DictConfig):
        self.n_frames = int(cfg.n_frames)
        self.n_rounds = int(cfg.n_rounds)
        self.long_side = int(cfg.long_side)
        self.short_side = int(cfg.short_side)
        self.max_new_tokens = int(cfg.max_new_tokens)

        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def load_image(self, rgb_path: str) -> Image.Image:
        img = Image.open(rgb_path).convert("RGB")
        w, h = img.size
        if w >= h:
            size = (self.long_side, self.short_side)
        else:
            size = (self.short_side, self.long_side)
        # 只做等比例缩放，宽高比不一致时直接报错
        assert w * size[1] == h * size[0], (
            f"Aspect ratio of {rgb_path} ({w}x{h}) differs from {size[0]}x{size[1]}"
        )
        return img.resize(size, Image.BICUBIC)

    def select_frame(
        self,
        prompt: str,
        target_object: str,
        functional_component: str,
        rgb_paths: List[str],
    ) -> Tuple[int, str]:
        """
        Returns the index (in rgb_paths) of the selected frame and the raw response.
        """
        content = [
            {
                "type": "text",
                "text": USER_PROMPT_HEADER.format(
                    prompt=prompt,
                    target_object=target_object,
                    functional_component=functional_component,
                    last_index=len(rgb_paths) - 1,
                ),
            }
        ]
        for i, rgb_path in enumerate(rgb_paths):
            content.append({"type": "text", "text": f"frame_{i}:"})
            content.append({"type": "image", "image": self.load_image(rgb_path)})
        content.append({"type": "text", "text": USER_PROMPT_QUERY})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return self.parse_response(response, len(rgb_paths)), response

    @staticmethod
    def parse_response(response: str, n_images: int) -> int:
        data = json.loads(response)
        assert isinstance(data, dict) and set(data.keys()) == {"frame"}, response
        labels = {f"frame_{i}": i for i in range(n_images)}
        assert data["frame"] in labels, response
        return labels[data["frame"]]

    def sample_rounds(self, n_video_frames: int) -> List[np.ndarray]:
        """
        K rounds of uniform temporal sampling with different start offsets, at most N frames each.
        Rounds with identical frame indices (short videos) are kept only once.
        """
        interval = n_video_frames / self.n_frames
        rounds = list()
        for r in range(self.n_rounds):
            start = r * interval / self.n_rounds
            positions = np.floor(start + np.arange(self.n_frames) * interval).astype(int)
            idxs = np.unique(positions[positions < n_video_frames])
            if not any(np.array_equal(idxs, prev) for prev in rounds):
                rounds.append(idxs)
        return rounds

    def search_video(
        self,
        prompt: str,
        target_object: str,
        functional_component: str,
        rgb_frames: Dict[str, str],
    ) -> dict:
        """
        Searches one video for the functional component.
        rgb_frames maps SceneFun3D frame ids (timestamps) to high resolution RGB paths.
        Returns the per-round retrieval record and the candidate frame ids.
        """
        frame_ids = sorted(rgb_frames.keys(), key=float)
        n_video_frames = len(frame_ids)
        interval = n_video_frames / self.n_frames
        radius = int(round(interval / 2))

        record = {
            "n_video_frames": n_video_frames,
            "interval": interval,
            "radius": radius,
            "rounds": list(),
        }
        candidate_idxs = set()
        for r, idxs in enumerate(self.sample_rounds(n_video_frames)):
            sampled_ids = [frame_ids[i] for i in idxs]
            selected, response = self.select_frame(
                prompt,
                target_object,
                functional_component,
                [rgb_frames[frame_id] for frame_id in sampled_ids],
            )

            center = int(idxs[selected])
            neighborhood = range(
                max(0, center - radius), min(n_video_frames - 1, center + radius) + 1
            )
            candidate_idxs.update(neighborhood)
            round_record = {
                "round": r,
                "sampled_frame_ids": sampled_ids,
                "response": response,
                "selected_label": f"frame_{selected}",
                "selected_frame_id": frame_ids[center],
                "neighborhood_frame_ids": [frame_ids[i] for i in neighborhood],
            }
            record["rounds"].append(round_record)
            print(f"  round {r}: {response.strip()} -> {round_record['selected_frame_id']}")

        record["candidate_frame_ids"] = [frame_ids[i] for i in sorted(candidate_idxs)]
        return record
