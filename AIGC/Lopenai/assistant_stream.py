import os
from typing_extensions import override
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, DefaultHttpxClient
from openai import AssistantEventHandler
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import (
    CodeInterpreterToolCall, CodeInterpreterToolCallDelta,
    FileSearchToolCall, FileSearchToolCallDelta,
    FunctionToolCall, FunctionToolCallDelta
)
streamflag = True
load_dotenv(find_dotenv(".env"), override=True)

# Step 1: Create an assistant
# An Assistant represents an entity that can be configured to respond to a user's messages
# using several parameters like model, instructions, and tools.
client = OpenAI(
    api_key=os.getenv("TRANSFER_KEY"),
    base_url=os.getenv("TRANSFER_URL") + "/v1/",
    http_client=DefaultHttpxClient(
        proxies="http://localhost:7890"
    )
)
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo"
)

# Step 2: Create a Thread
# A Thread represents a conversation between a user and one or many Assistants.
# You can create a Thread when a user (or your AI application) starts a conversation with your Assistant.
thread = client.beta.threads.create()

# Step 3: Add Message to the 
# The contents of the messages your users or applications create are added as Message objects to the Thread. 
# Messages can contain both text and files.
# There is no limit to the number of Messages you can add to Threads
# we smartly truncate any context that does not fit into the model's context window.
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

# Step create a run
# Once all the user Messages have been added to the Thread, you can Run the Thread with any Assistant.
# Creating a Run uses the model and tools associated with the Assistant to generate a response.
# These responses are added to the Thread as assistant Messages.
if not streamflag:
    run = client.beta.threads.create_and_run_poll(
        thread=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )  # already run

    # list the Messages added to the Thread by the Assistant
    if run.status == "completed":
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        print(messages)
    else:
        print(run.status)
else:
    # First, we create a EventHandler class to define
    # how we want to handle the events in the response stream.
    class EventHandler(AssistantEventHandler):
        @override
        def on_text_created(self, text: Text) -> None:
            print(f"\nassistant > ", end="", flush=True)

        def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
            print(delta.value, end="", flush=True)

        def on_tool_call_created(self, tool_call: CodeInterpreterToolCall | FileSearchToolCall | FunctionToolCall) -> None:
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        def on_tool_call_delta(
            self,
            delta: CodeInterpreterToolCallDelta | FileSearchToolCallDelta | FunctionToolCallDelta,
            snapshot: CodeInterpreterToolCall | FileSearchToolCall | FunctionToolCall
        ) -> None:
            if delta.type == "code_interpreter":
                if delta.code_interpreter.input:
                    print(delta.code_interpreter.input, end="", flush=True)
                if delta.code_interpreter.outputs:
                    print(f"\n\noutput >", flush=True)
                    for output in delta.code_interpreter.outputs:
                        if output.type == "logs":
                            print(f"\n{output.logs}", flush=True)

    # Then, we use the `stream` SDK helper 
    # with the `EventHandler` class to create the Run 
    # and stream the response.
    with client.beta.threads.create_and_run_stream(
        thread=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler()
    ) as stream:
        stream.until_done()