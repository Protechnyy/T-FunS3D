import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig

import json
import time

COMPONENT_TARGET_RELATIONS = ("attached", "remote")

class LLMParser:
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-14B",
                 torch_dtype: str = "auto",
                 device_map: str = "auto",
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only = False,
        )


        self.system_prompt1 = """You are an AI system that generates JSON instructions for a robot to identify physical objects and their spatial relations based on a natural language command.

- Use only concrete physical objects (furniture, appliances, manipulable items).
- Do NOT use rooms (e.g., kitchen, bedroom) as objects or in relations.
- Retain descriptive adjectives of objects (e.g., "blue chair", "wooden desk").
- The only allowed spatial relations are: [in, on, under, around, next to, near, over, behind, in_front_of, to_the_left, to_the_right].
- Relations must be between visible or manipulable physical objects only.
- Extract only spatial relations explicitly expressed in the task description.
- Do not infer uncertain spatial relations.
- referent_objects must contain the physical objects participating in the extracted spatial relations and used for locating the task-relevant object.
- Keep referent_objects in the order implied by the spatial relations.
- Output valid JSON only with no text before or after.
- JSON rules: double quotes for keys and strings, lowercase true/false, null if no value, lists as JSON arrays."""

        self.system_prompt2 = """You are an AI system that generates JSON instructions for a robot to understand interactions with physical objects.

- Retain adjectives describing object properties when they are needed for identification, e.g. "red button", "right drawer", "left door".
- The only allowed robot actions are: [rotate, key_press, tip_push, hook_pull, pinch_pull, hook_turn, foot_push, plug_in, unplug].

- target_object is the specific physical object whose state or function is intended to change as a result of the task.
  Examples:
  - "open the bottom drawer" -> "bottom drawer"
  - "open the cabinet door" -> "cabinet door"
  - "turn on the ceiling light" -> "ceiling light"

- target_object_hierarchy describes part-of relations from the top-level scene object to target_object.
  The last element must be target_object.
  Examples:
  - ["cabinet", "bottom drawer"]
  - ["closet", "left door"]
  - ["ceiling light"]
  Do NOT include referent objects in target_object_hierarchy.
  Do NOT include functional_component in target_object_hierarchy.

- functional_component is the lowest-level physical component that the robot must directly manipulate to execute the task.
  Examples include handle, knob, button, switch, latch, plug, and dial.
  Infer the functional component when it is required for the task even if it is not explicitly named.

- component_target_relation must be exactly one of ["attached", "remote"].

- Use "attached" when functional_component is part of, mounted on, or directly physically connected to target_object or an object in target_object_hierarchy.

- Use "remote" when functional_component is physically separate from target_object and controls or affects it through a functional relationship.
  Example: a wall light switch controlling a ceiling light has a remote relation.

- Output valid JSON only with no text before or after.
- JSON rules: double quotes for keys and strings, lowercase true/false, null if no value, lists as JSON arrays."""

        self.statement1 = """To {query}, what spatial information do I know?

Respond with valid JSON only in the following format:

{{
  "prompt": "{query}",
  "spatial_relations": [
    "Spatial relations explicitly expressed between physical objects, e.g. 'TV is on top of the cabinet'."
  ],
  "referent_objects": [
    "Physical objects participating in the spatial relations and used to locate the task-relevant object."
  ]
}}

If no spatial relations exist, return an empty list for spatial_relations.
If no referent objects exist, return an empty list for referent_objects."""

        self.statement2 = """I know the following spatial information:

{spatial_relations}

Analyze this robot task:

"{query}"

Respond with valid JSON only in the following format:

{{
  "prompt": "{query}",
  "target_object": "the specific physical object whose state or function is intended to change",
  "target_object_hierarchy": [
    "A part-of hierarchy from the top-level scene object to target_object."
  ],
  "functional_component": "the lowest-level physical component that must be directly manipulated",
  "component_target_relation": "attached or remote"
}}"""


    def parse_query(self, query: str, max_new_tokens: int = 32768, show_thinking: bool = False) -> dict:

        messages = [
            {"role": "system", "content": self.system_prompt1},
            {"role": "user", "content": self.statement1.format(query=query.lower())}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        start = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        stop = time.time()
        print(f"System respond in {stop - start:.2f}s.")

        spatial_content = json.loads(self.show_results(output_ids))
        self.validate_spatial_content(spatial_content)
        self.content = {"prompt": spatial_content.pop("prompt")}
        self.content.update(spatial_content)

        # 第二轮是独立对话，第一轮结果只通过 statement2 中的 spatial information 传入
        messages = [
            {"role": "system", "content": self.system_prompt2},
            {"role": "user", "content": self.statement2.format(spatial_relations=json.dumps(spatial_content, indent=2), query=query.lower())}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        start = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        stop = time.time()
        print(f"System respond in {stop - start:.2f}s.")

        interaction_content = json.loads(self.show_results(output_ids))
        self.validate_interaction_content(interaction_content)
        interaction_content.pop("prompt")
        self.content.update(interaction_content)
        print(f"[INFO] Completed query parsing.")

        return self.content

    @staticmethod
    def validate_spatial_content(content: dict):
        assert set(content.keys()) == {"prompt", "spatial_relations", "referent_objects"}, content
        assert isinstance(content["spatial_relations"], list), content
        assert all(isinstance(rel, str) for rel in content["spatial_relations"]), content
        assert isinstance(content["referent_objects"], list), content
        assert all(isinstance(obj, str) for obj in content["referent_objects"]), content

    @staticmethod
    def validate_interaction_content(content: dict):
        assert set(content.keys()) == {
            "prompt",
            "target_object",
            "target_object_hierarchy",
            "functional_component",
            "component_target_relation",
        }, content
        assert isinstance(content["target_object"], str), content
        assert isinstance(content["target_object_hierarchy"], list), content
        assert len(content["target_object_hierarchy"]) > 0, content
        assert all(isinstance(obj, str) for obj in content["target_object_hierarchy"]), content
        assert isinstance(content["functional_component"], str), content
        assert content["component_target_relation"] in COMPONENT_TARGET_RELATIONS, content

    def show_results(self, output_ids, show_thinking=False):
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        if show_thinking:
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            print("thinking content:", thinking_content)

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("response:", content)
        return content

    def get_target_object(self) -> str:
        """
        Get the target object from the query using the LLM.
        """
        return self.content["target_object"]

    def get_target_object_hierarchy(self) -> list:
        """
        Get the target object hierarchy from the query using the LLM.
        """
        return self.content["target_object_hierarchy"]

    def get_functional_component(self) -> str:
        """
        Get the functional component from the query using the LLM.
        """
        return self.content["functional_component"]

    def get_component_target_relation(self) -> str:
        """
        Get the relation between the functional component and the target object.
        """
        return self.content["component_target_relation"]

    def get_referent_objects(self) -> list:
        """
        Get the referent objects from the query using the LLM.
        """
        return self.content["referent_objects"]

    def get_spatial_relations(self) -> list:
        """
        Get the spatial relations from the query using the LLM.
        """
        return self.content["spatial_relations"]
