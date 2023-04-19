# Editor Extensions

To use Tabby as your code assistant, you can install the Tabby extension for the following editors:

* [VSCode Extension](../../clients/vscode)
* [VIM Extension](../../clients/vim)

If you use other editors, you can build your own editor extension as a Tabby client.

## Build a HTTP Client

Tabby server exposes HTTP API using [FastAPI](https://github.com/tiangolo/fastapi), based on  OpenAPI standards. If you followed the instructions and are running Tabby Server locally, you can find the API documentation at [localhost:5000](http://localhost:5000), otherwise you can find an online version at [tabbyml.github.io/tabby](https://tabbyml.github.io/tabby). You can also get a JSON schema at [localhost:5000/openapi.json](http://localhost:5000/openapi.json), or a static file version here: [openapi.json](../../docs/openapi.json).

We suggest you to use OpenAPI Generators to generate a client for your language. For example, to generate a TypeScript client, you can use [openapi-typescript-codegen](https://github.com/ferdikoomen/openapi-typescript-codegen):

```bash
# Install openapi-typescript-codegen package
yarn global add openapi-typescript-codegen
# Generate a client using axios
openapi --input ../../docs/openapi.json --output ./src/generated --client axios --name Tabby
```

You can also refer to the [packages.json](../../clients/vscode/packages.json) file in VSCode Extension directory to find how to set up code generation.

## Trigger a Completion

If your editor uses a completion engine, building a completion provider for the engine can be a good choice. For example, in VSCode Extension, we implement an `InlineCompletionItemProvider` and register it to the host context. The VSCode host will call the completion provider when the user is typing or manually triggers completion using hotkeys.
In the VIM Extension, we don't depend on other completion engines, so we just listen to `CursorMovedI` events and schedule a completion request.

### Make a Completion Request

To make a completion request, send a POST request to [`/v1/completions`](http://localhost:5000/#/default/completions_v1_completions_post). The request body should contains two fields:
* `prompt`: the code snippet
  All text before the cursor should be included in the prompt. However, to control the request body size, we suggest setting a maximum number of lines in the prompt, for example, 20 lines.
* `language`: the language of the code snippet
  Specify the language of the code snippet, such as `python`, `typescript`, `javascript`, etc. We use [VSCode language identifiers](https://code.visualstudio.com/docs/languages/identifiers) as standards. You may need to convert other editor's file types to this standard. For example, in the VIM Extension, we use a `g:tabby_filetype_to_languages` global dictionary to map VIM's file types to VSCode language identifiers.

### Parse the Completion Response

The response body contains the following fields:
* `id`: a string identifier of the completion, used to track completion events
* `created`: a numeric timestamp when the completion was created
* `choices`: a list of `Choice` objects, maybe empty if code language is not supported, or if no completion is provided

`Choice` contains the following fields:
* `index`: a numberic index of the choice, also used to track completion events
* `text`: the string to be inserted after cursor

If you are building a provider for a completion engine, format this response to the engine's completion item format.
Otherwise, you may want to build your own UI to show the completion choices, and listen to a hotkey for text insertion. For example, you can show completion as ghost text and use the `<TAB>` key to insert text, as we do in the VIM Extension.
Note that you may want to delete the trailing parentheses when inserting the completion text.

## Send Feedback Event to Tabby Server

Feedback to the Tabby Server is important to improve completion quality. We should send feedback events when:
* A completion choice appears on UI (type: `view`)
* User accpect a completion choice (type: `select`)

### Make an Event Request

To make an event request, send a POST request to [`/v1/events`](http://localhost:5000/#/default/events_v1_events_post). The request body should contains three fields:
* `type`: the type of the event, `view` or `select`
* `completion_id`: the id of the completion, should be the same as the `id` field in the completion response
* `choice_index`: the index of the choice, should be the same as the `index` field of choice item in the completion response

The response contains only a string `ok` which you can simply ignore.
