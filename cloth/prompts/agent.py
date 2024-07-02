SYSTEM_PROMPT = """\
You are a personal assistant that is tasked with helping the user answer questions based on a knowledge graph.

Here is a summary of the relevant information from the knowledge graph:
```
{information}
```
Based on the information given to you, answer the user's query. Answer in markdown.\
"""

MAP_PROMPT = """You will be summarizing the key relations about a specific node in a knowledge graph. 
I will provide you with the node and a list of incoming and outgoing relations involving that node. 
Each relation will be represented as a source, link, target triple.

Here is the {type} the relations will be centered around:
<{type}>
{name}
</{type}>

And here are the relations for this {type}:
<relations>
{docs}
</relations>
Please carefully review the provided relations, analyzing the key connections, properties and relationships they reveal about the central node. /
Please write a concise summary of the key relations that captures the most essential information about the node. /
Focus on the relations that are most defining, relevant and important for understanding what this node represents and how it is connected in the knowledge graph. /
Try to cover the key points in 2-4 sentences if possible."""

REDUCE_PROMPT = """You are given a set of summaries that were extracted from a knowledge graph within the delimiters:
```
{docs}
```
Center your summary around this {type}: {name}
Based on the summaries, provide a concise summary. \
Respond with ONLY the summary and nothing else. Start your response with the summary:"""