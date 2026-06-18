# Agents, Tool-Calling, and the Model Context Protocol

## What an agent is

An agent is a system in which a language model decides which actions to take, rather than
following a fixed script. Given a goal and a set of tools, the model chooses what to do,
executes it through code, observes the result, and decides again, looping until the goal is
met. This decision-making loop is what distinguishes an agent from a static pipeline.

## Tool-calling

Tool-calling is the mechanism that lets a model use external functions. The critical point,
often misunderstood, is that the model does not run any code itself. It emits a structured
request naming a tool and its arguments. The surrounding application code executes the
corresponding function and returns the result to the model, which then continues. The model
decides; the application executes.

A model is told about its tools through their names, parameters, and descriptions. These
descriptions are the model's only guide to when and how to use each tool, so they must be
written carefully. Tools exist because models cannot reliably do certain things on their own:
exact arithmetic, looking up live or private data, and taking real actions such as sending a
message or querying a database.

## LangGraph and agent frameworks

Building the reason-act-observe loop by hand is tedious, so frameworks automate it. LangGraph
models an agent as a graph whose state is the growing list of messages. Nodes do work, such
as calling the model or executing a tool, and edges, including conditional edges, route
between them. A typical agent graph has an agent node that calls the model and a tool node
that executes requested tools, with a conditional edge that loops back to the agent after a
tool runs and exits when the model produces a final answer. This is the ReAct loop expressed
as a graph.

## The Model Context Protocol

The Model Context Protocol, or MCP, is an open standard that lets tools live in a standalone
server which any compatible client can connect to and use. Before MCP, every application had
to integrate each tool directly. MCP decouples tools from applications, so a tool written
once can be reused by many hosts, including chat applications, development environments, and
custom agents. It has been widely described as a universal connector for AI tools.

An MCP system has three roles. The host is the AI application. It contains a client, which is
the connector that speaks the protocol. The client connects to a server, which exposes
capabilities. Servers expose three kinds of primitive: tools, which are functions the model
can call; resources, which are data the host can read, such as files or database records; and
prompts, which are reusable templates. The protocol is built on JSON-RPC and supports two
main transports: standard input and output for a local server run as a subprocess, and
streamable HTTP for a remote server reached over the network.

The practical benefit is reuse and interoperability. By exposing your tools through an MCP
server, you make them available to the entire ecosystem of MCP-compatible clients rather than
locking them inside a single application. Conversely, by acting as an MCP client, your agent
can immediately use the growing library of existing MCP servers.
