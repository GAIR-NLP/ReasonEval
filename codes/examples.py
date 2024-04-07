import argparse
import torch
import torch.nn as nn
from transformers import MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig


class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores
    
class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
      
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores

def get_results(args, question, reasoning_steps):
    # Loading the model
    model_file_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    if args.model_size == '7B':
        model = ReasonEval_7B.from_pretrained(model_file_path)
    elif args.model_size == '34B':
        model = ReasonEval_34B.from_pretrained(model_file_path)
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")

    # Preprocessing input
    PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    step_separator = f"{tokenizer.pad_token}"
    combined_steps = ""
    for steps in reasoning_steps:
        combined_steps += steps + step_separator
    prompt = PROMPT_FORMAT.format(input=question)
    tokenized_result = tokenizer(prompt + step_separator + combined_steps)['input_ids']

    ## Separating labels and adjusting token IDs
    separator_token_id = tokenizer(step_separator)['input_ids'][-1]
    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0
    for idx, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(idx - 1 - separator_count)
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)
    if args.model_size == '7B':
        adjusted_token_ids = [1] + adjusted_token_ids # Adjusting to recover the first token_ids of the sentences
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[2:]  # Adjusting to skip the first two separator (begining and endding of the problems)
    elif args.model_size == '34B':
        adjusted_token_ids=torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[1:]  # Adjusting to skip the first separator (endding of the problems)
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")
    assert len(labeled_token_indices) == len(reasoning_steps)

    attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
    # Evaluating reasoning steps using ReasonEval
    with torch.no_grad():
        reasoning_scores = model(adjusted_token_ids, attention_mask)[0,labeled_token_indices , :]
        scores = torch.softmax(reasoning_scores, dim=-1).tolist()

    # Calculating the validity and redundancy scores
    ## score: [p_{negative}, p_{neutral}, p_{positive}]

    ## S_{validity} = p_{neutral} + p_{positive}
    step_level_validity_scores =  [(score[1] + score[2]) for score in scores]
    print(f"step_level_validity_scores: {step_level_validity_scores}")
    # ReasonEval-7B for the example: [0.9492, 0.7863 ,0.2520, 0.7860, 0.9125, 0.5916, 0.4494, 0.8189, 0.8240, 0.2671]
    # ReaspnEval-34B for the example: [0.9360, 0.6813 ,0.0720, 0.2811, 0.4531, 0.1122, 0.1328, 0.2026, 0.2265 0.1163]

    ## S_{redundancy} = p_{neutral}
    step_level_redundancy_scores = [score[1] for score in scores]
    print(f"step_level_redundancy_scores: {step_level_redundancy_scores}")
    # ReasonEval-7B for the example: [0.4433, 0.1287, 0.0397, 0.0789, 0.0789, 0.0509, 0.0487, 0.0702, 0.0955, 0.0120]
    # ReasonEval-34B for the example: [0.6060, 0.1682, 0.0258, 0.1044, 0.1604, 0.0404, 0.0447, 0.0507, 0.0492, 0.0236]
    
    solution_level_validity_scores = min(step_level_validity_scores)
    print(f"solution_level_validity_scores: {solution_level_validity_scores}")
    solution_level_redundancy_scores = max(step_level_redundancy_scores)
    print(f"solution_level_validity_scores: {solution_level_redundancy_scores}")


if __name__ == "__main__":
    question = "Let $x,$ $y,$ and $z$ be positive real numbers such that $xyz(x + y + z) = 1.$  Find the minimum value of\n\\[(x + y)(y + z).\\]"
    ### The solution reaches the correct result, but there is redundancy in step 1, and the first invalid step occurs in step 3.
    reasoning_steps = ["1. The problem asks us to find the minimum value of $(x + y)(y + z)$ given that $x,$ $y,$ and $z$ are positive real numbers and $xyz(x + y + z) = 1$.", 
    "2. By the AM-GM inequality, we have $x + y + z \\geq 3\\sqrt[3]{xyz}$.", 
    "3. By the given condition $xyz(x + y + z) = 1$, we can substitute $x + y + z$ with $\\sqrt[3]{xyz}$ in the inequality from step 2 to get $3\\sqrt[3]{xyz} \\geq 3$.",
    "4. Simplifying the inequality from step 3 gives $\\sqrt[3]{xyz} \\geq 1$.",
    "5. By raising both sides of the inequality from step 4 to the power of 3, we have $xyz \\geq 1$.",
    "6. By the AM-GM inequality, we have $(x + y)(y + z) \\geq 2\\sqrt{(x + y)(y + z)}$.",
    "7. By the given condition $xyz(x + y + z) = 1$, we can substitute $(x + y)(y + z)$ with $\\frac{1}{xyz}$ in the inequality from step 6 to get $2\\sqrt{(x + y)(y + z)} \\geq 2\\sqrt{\\frac{1}{xyz}}$.", 
    "8. Simplifying the inequality from step 7 gives $(x + y)(y + z) \\geq \\frac{2}{\\sqrt{xyz}}$.", 
    "9. By the condition $xyz \\geq 1$ from step 5, we have $\\frac{2}{\\sqrt{xyz}} \\geq \\frac{2}{\\sqrt{1}} = 2$.", 
    "10. Therefore, the minimum value of $(x + y)(y + z)$ is $\\boxed{2}$."]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,default='GAIR/ReasonEval-7B')
    parser.add_argument("--model_size", type=str, choices=['7B', '34B'], default='7B')
    args = parser.parse_args()
    get_results(args, question, reasoning_steps)
