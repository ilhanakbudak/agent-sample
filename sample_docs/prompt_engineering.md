# Prompt Engineering Techniques

Prompt engineering is the practice of designing the input given to a language model so that
it produces the desired output reliably. Because a model's behavior is shaped almost entirely
by its prompt, small changes in wording can produce large changes in quality.

## Zero-shot and few-shot prompting

Zero-shot prompting asks the model to perform a task with only an instruction and no
examples. It works well for tasks the model understands inherently. Few-shot prompting
includes several worked examples in the prompt, demonstrating the desired input-output
pattern. Examples are powerful because they show the model the exact format and style
expected, and they often outperform lengthy instructions.

## Chain-of-thought

Chain-of-thought prompting asks the model to reason step by step before giving a final
answer. For tasks involving arithmetic, logic, or multi-step reasoning, encouraging
intermediate steps substantially improves accuracy, because the model allocates more
computation to the problem and is less likely to leap to a wrong conclusion.

## System prompts and roles

Most chat models separate a system message from user messages. The system prompt sets the
model's persistent role, constraints, and policy. It is the right place to specify tone,
output format, safety rules, and, in an agent, the policy governing when to use tools. A
well-crafted system prompt removes ambiguity and reduces the model's tendency to fall back
on undesired default behavior.

## ReAct and tool use

The ReAct pattern interleaves reasoning and acting. The model reasons about what it needs,
acts by calling a tool, observes the result, and continues. This is the foundation of agents.
The instructions for each tool, especially its description, function as prompts: a vague tool
description leads the model to use the tool incorrectly or not at all.

## Controlling output

Several levers control output. Temperature governs randomness: low values near zero make the
output deterministic and focused, which suits factual extraction, while higher values increase
creativity and variety. Explicitly specifying the output format, such as requesting JSON or a
particular structure, makes outputs easier to parse downstream. Asking the model to refuse or
to say it does not know when information is missing is essential for grounded systems.

## Common pitfalls

Overly long prompts can bury the important instruction; place critical instructions
prominently. Conflicting instructions confuse the model. Relying on the model to infer an
unstated requirement is fragile; state requirements explicitly. And prompts that work with a
large, capable model may fail with a smaller one, which follows instructions less reliably and
often needs more forceful, unambiguous wording.
