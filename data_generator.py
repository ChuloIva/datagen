"""
Main Data Generation Engine for Cognitive Action Training Data
Integrates with Ollama for LLM generation
"""

import json
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from variable_pools import COGNITIVE_ACTIONS, get_random_selection
from prompt_templates import generate_prompt, generate_batch_prompts

@dataclass
class GeneratedExample:
    """Structure for a generated training example"""
    text: str
    primary_cognitive_action: str
    secondary_actions: List[str]
    domain: str
    complexity: str
    perspective: str
    format_type: str
    metadata: Dict[str, Any]

class CognitiveDataGenerator:
    """Main class for generating cognitive action training data"""

    def __init__(self, ollama_client=None):
        """Initialize the generator with optional Ollama client"""
        self.ollama_client = ollama_client
        self.generated_examples = []
        self.generation_stats = {
            'total_generated': 0,
            'by_cognitive_action': {},
            'by_domain': {},
            'by_complexity': {},
            'errors': []
        }

    def generate_single_example(self,
                               cognitive_action: Optional[str] = None,
                               template_type: str = "single",
                               model: str = "llama3.2") -> Optional[GeneratedExample]:
        """Generate a single training example"""
        try:
            # Generate prompt
            prompt, params = generate_prompt(cognitive_action, template_type,
                                           self.generation_stats['total_generated'])

            # Generate with Ollama
            if self.ollama_client:
                response = self.ollama_client.generate(
                    model=model,
                    prompt=prompt
                )
                generated_text = response['response'].strip()
            else:
                # Fallback for testing without Ollama
                generated_text = f"[Generated example for {params['cognitive_action']} in {params['domain']}]"

            # Create example object
            example = GeneratedExample(
                text=generated_text,
                primary_cognitive_action=params['cognitive_action'],
                secondary_actions=[],  # Could be extracted with NLP
                domain=params['domain'],
                complexity=params['complexity_level'],
                perspective=params['perspective'],
                format_type=template_type,
                metadata={
                    'subject': params['subject'],
                    'emotional_state': params['emotional_state'],
                    'language_style': params['language_style'],
                    'unique_angle': params['unique_angle'],
                    'trigger': params['trigger'],
                    'generation_timestamp': time.time(),
                    'prompt_used': prompt
                }
            )

            # Update statistics
            self._update_stats(example)
            self.generated_examples.append(example)

            return example

        except Exception as e:
            error_info = {
                'error': str(e),
                'cognitive_action': cognitive_action,
                'template_type': template_type,
                'timestamp': time.time()
            }
            self.generation_stats['errors'].append(error_info)
            print(f"Error generating example: {e}")
            return None

    def generate_batch(self,
                      batch_size: int = 10,
                      cognitive_action: Optional[str] = None,
                      template_type: str = "single",
                      model: str = "llama3.2",
                      delay: float = 0.1) -> List[GeneratedExample]:
        """Generate a batch of examples"""
        examples = []

        for i in range(batch_size):
            example = self.generate_single_example(cognitive_action, template_type, model)
            if example:
                examples.append(example)
                print(f"Generated example {i+1}/{batch_size}: {example.primary_cognitive_action}")

            # Small delay to avoid overwhelming the API
            time.sleep(delay)

        return examples

    def generate_stratified_dataset(self,
                                   total_examples: int = 1000,
                                   model: str = "llama3.2") -> List[GeneratedExample]:
        """Generate a stratified dataset ensuring coverage across cognitive actions"""
        examples_per_action = total_examples // len(COGNITIVE_ACTIONS)
        all_examples = []

        for action in COGNITIVE_ACTIONS.keys():
            print(f"Generating {examples_per_action} examples for: {action}")

            # Mix template types for variety
            template_types = ["single"] * int(examples_per_action * 0.7) + \
                           ["chain"] * int(examples_per_action * 0.2) + \
                           ["dialogue"] * int(examples_per_action * 0.1)

            random.shuffle(template_types)

            for i, template_type in enumerate(template_types):
                example = self.generate_single_example(action, template_type, model)
                if example:
                    all_examples.append(example)

        # Shuffle final dataset
        random.shuffle(all_examples)
        return all_examples

    def generate_phase_dataset(self,
                              phase: str = "phase1",
                              target_examples: int = 20000,
                              model: str = "llama3.2") -> List[GeneratedExample]:
        """Generate dataset according to specific phases from instructions"""

        if phase == "phase1_round1":
            # Core cognitive actions - 500 examples each
            return self._generate_phase1_round1(target_examples, model)

        elif phase == "phase1_round2":
            # Cognitive action combinations
            return self._generate_phase1_round2(target_examples, model)

        elif phase == "phase2":
            # Domain variation
            return self._generate_phase2(target_examples, model)

        elif phase == "phase3":
            # Complexity and perspective variation
            return self._generate_phase3(target_examples, model)

        elif phase == "phase4":
            # Negative examples
            return self._generate_phase4(target_examples, model)

        elif phase == "phase5":
            # Dialogue format
            return self._generate_phase5(target_examples, model)

        elif phase == "phase6":
            # Thought-stream format
            return self._generate_phase6(target_examples, model)

        else:
            # Default to stratified
            return self.generate_stratified_dataset(target_examples, model)

    def _generate_phase1_round1(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 1 Round 1: Core cognitive actions"""
        examples_per_action = target_examples // len(COGNITIVE_ACTIONS)
        all_examples = []

        for action in COGNITIVE_ACTIONS.keys():
            examples = self.generate_batch(examples_per_action, action, "single", model)
            all_examples.extend(examples)

        return all_examples

    def _generate_phase1_round2(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 1 Round 2: Cognitive action combinations"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "chain", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase2(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 2: Domain variation"""
        # Take seed examples and vary domains
        return self.generate_stratified_dataset(target_examples, model)

    def _generate_phase3(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 3: Complexity and perspective variation"""
        all_examples = []

        # Mix of complexity levels and perspectives
        for i in range(target_examples):
            template_type = random.choice(["single", "chain", "thought_stream"])
            example = self.generate_single_example(None, template_type, model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase4(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 4: Negative examples"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "negative", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase5(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 5: Dialogue format"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "dialogue", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _generate_phase6(self, target_examples: int, model: str) -> List[GeneratedExample]:
        """Phase 6: Thought-stream format"""
        all_examples = []

        for i in range(target_examples):
            example = self.generate_single_example(None, "thought_stream", model)
            if example:
                all_examples.append(example)

        return all_examples

    def _update_stats(self, example: GeneratedExample):
        """Update generation statistics"""
        self.generation_stats['total_generated'] += 1

        # Update by cognitive action
        action = example.primary_cognitive_action
        if action not in self.generation_stats['by_cognitive_action']:
            self.generation_stats['by_cognitive_action'][action] = 0
        self.generation_stats['by_cognitive_action'][action] += 1

        # Update by domain
        domain = example.domain
        if domain not in self.generation_stats['by_domain']:
            self.generation_stats['by_domain'][domain] = 0
        self.generation_stats['by_domain'][domain] += 1

        # Update by complexity
        complexity = example.complexity
        if complexity not in self.generation_stats['by_complexity']:
            self.generation_stats['by_complexity'][complexity] = 0
        self.generation_stats['by_complexity'][complexity] += 1

    def export_dataset(self, filepath: str, format: str = "jsonl"):
        """Export generated dataset to file"""
        if format == "jsonl":
            with open(filepath, 'w') as f:
                for example in self.generated_examples:
                    json_obj = {
                        'text': example.text,
                        'primary_cognitive_action': example.primary_cognitive_action,
                        'secondary_actions': example.secondary_actions,
                        'domain': example.domain,
                        'complexity': example.complexity,
                        'perspective': example.perspective,
                        'format_type': example.format_type,
                        'metadata': example.metadata
                    }
                    f.write(json.dumps(json_obj) + '\n')

        elif format == "json":
            with open(filepath, 'w') as f:
                dataset = [
                    {
                        'text': example.text,
                        'primary_cognitive_action': example.primary_cognitive_action,
                        'secondary_actions': example.secondary_actions,
                        'domain': example.domain,
                        'complexity': example.complexity,
                        'perspective': example.perspective,
                        'format_type': example.format_type,
                        'metadata': example.metadata
                    }
                    for example in self.generated_examples
                ]
                json.dump(dataset, f, indent=2)

        print(f"Exported {len(self.generated_examples)} examples to {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats

    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.generation_stats
        print(f"\nGeneration Statistics:")
        print(f"Total examples generated: {stats['total_generated']}")
        print(f"Errors encountered: {len(stats['errors'])}")

        print(f"\nBy Cognitive Action:")
        for action, count in sorted(stats['by_cognitive_action'].items()):
            print(f"  {action}: {count}")

        print(f"\nBy Domain:")
        for domain, count in sorted(stats['by_domain'].items()):
            print(f"  {domain}: {count}")

        print(f"\nBy Complexity:")
        for complexity, count in sorted(stats['by_complexity'].items()):
            print(f"  {complexity}: {count}")