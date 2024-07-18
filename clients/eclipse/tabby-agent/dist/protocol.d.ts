import { MessageDirection, ProtocolRequestType, InitializeError, InitializeParams as InitializeParams$1, InitializeResult as InitializeResult$1, ClientCapabilities as ClientCapabilities$1, ServerCapabilities as ServerCapabilities$1, RegistrationType, ProtocolNotificationType, DidChangeConfigurationParams as DidChangeConfigurationParams$1, CodeLensParams, CodeLens as CodeLens$1, Command, CompletionParams, CompletionList as CompletionList$1, CompletionItem as CompletionItem$1, InlineCompletionParams, InlineCompletionList as InlineCompletionList$1, InlineCompletionItem as InlineCompletionItem$1, Location, URI, ProtocolRequestType0, Range, DeclarationParams, Declaration, LocationLink, SemanticTokensRangeParams, SemanticTokensLegend, SemanticTokens } from 'vscode-languageserver-protocol';

/**
 * Extends LSP method Initialize Request(↩️)
 *
 * - method: `initialize`
 * - params: {@link InitializeParams}
 * - result: {@link InitializeResult}
 */
declare namespace InitializeRequest {
    const method: "initialize";
    const messageDirection: MessageDirection;
    const type: ProtocolRequestType<InitializeParams, InitializeResult, InitializeError, void, void>;
}
type InitializeParams = InitializeParams$1 & {
    clientInfo?: ClientInfo;
    initializationOptions?: {
        config?: ClientProvidedConfig;
    };
    capabilities: ClientCapabilities;
};
type InitializeResult = InitializeResult$1 & {
    capabilities: ServerCapabilities;
};
/**
 * [Tabby] Defines the name and version information of the IDE and the tabby plugin.
 */
type ClientInfo = {
    name: string;
    version?: string;
    tabbyPlugin?: {
        name: string;
        version?: string;
    };
};
type ClientCapabilities = ClientCapabilities$1 & {
    tabby?: {
        /**
         * The client supports:
         * - `tabby/agent/didUpdateServerInfo`
         * - `tabby/agent/didChangeStatus`
         * - `tabby/agent/didUpdateIssues`
         * This capability indicates that client support receiving agent notifications.
         */
        agent?: boolean;
        /**
         * The client supports:
         * - `tabby/workspaceFileSystem/readFile`
         * This capability improves the workspace code snippets context (RAG).
         * When not provided, the server will try to fallback to NodeJS provided `fs` module,
         *  which is not available in the browser.
         */
        workspaceFileSystem?: boolean;
        /**
         * The client supports:
         * - `tabby/dataStore/get`
         * - `tabby/dataStore/set`
         * When not provided, the server will try to fallback to the default data store,
         *  a file-based data store (~/.tabby-client/agent/data.json), which is not available in the browser.
         */
        dataStore?: boolean;
        /**
         * The client supports:
         * - `tabby/languageSupport/textDocument/declaration`
         * - `tabby/languageSupport/textDocument/semanticTokens/range`
         * This capability improves the workspace code snippets context (RAG).
         */
        languageSupport?: boolean;
        /**
         * The client supports:
         * - `tabby/git/repository`
         * - `tabby/git/diff`
         * This capability improves the workspace git repository context (RAG).
         * When not provided, the server will try to fallback to the default git provider,
         *  which running system `git` command, not available if cannot execute `git` command,
         *  not available in the browser.
         */
        gitProvider?: boolean;
        /**
         * The client supports:
         * - `tabby/editorOptions`
         * This capability improves the completion formatting.
         */
        editorOptions?: boolean;
    };
};
type ServerCapabilities = ServerCapabilities$1 & {
    tabby?: {
        /**
         * The server supports:
         * - `tabby/chat/edit`
         * - `tabby/chat/generateCommitMessage`
         * See {@link ChatFeatureRegistration}
         */
        chat?: boolean;
    };
};
declare namespace ChatFeatureRegistration {
    const type: RegistrationType<unknown>;
}
/**
 * Extends LSP method Configuration Request(↪️)
 *
 * - method: `workspace/configuration`
 * - params: any
 * - result: {@link ClientProvidedConfig}[] (the array contains only one config)
 */
declare namespace ConfigurationRequest {
    const method: "workspace/configuration";
    const messageDirection: MessageDirection;
    const type: ProtocolRequestType<any, ClientProvidedConfig[], never, void, void>;
}
/**
 * [Tabby] Defines the config supported to be changed on the client side (IDE).
 */
type ClientProvidedConfig = {
    /**
     * Specifies the endpoint and token for connecting to the Tabby server.
     */
    server?: {
        endpoint?: string;
        token?: string;
    };
    /**
     * Trigger mode should be implemented on the client side.
     * Sending this config to the server is for telemetry purposes.
     */
    inlineCompletion?: {
        triggerMode?: "auto" | "manual";
    };
    /**
     * Keybindings should be implemented on the client side.
     * Sending this config to the server is for telemetry purposes.
     */
    keybindings?: "default" | "tabby-style" | "customize";
    /**
     * Controls whether the telemetry is enabled or not.
     */
    anonymousUsageTracking?: {
        disable?: boolean;
    };
};
/**
 * Extends LSP method DidChangeConfiguration Notification(➡️)
 * - method: `workspace/didChangeConfiguration`
 * - params: {@link DidChangeConfigurationParams}
 * - result: void
 */
declare namespace DidChangeConfigurationNotification {
    const method: "workspace/didChangeConfiguration";
    const messageDirection: MessageDirection;
    const type: ProtocolNotificationType<DidChangeConfigurationParams, void>;
}
type DidChangeConfigurationParams = DidChangeConfigurationParams$1 & {
    settings?: ClientProvidedConfig;
};
/**
 * Extends LSP method Code Lens Request(↩️)
 *
 * Tabby provides this method for preview changes applied in the Chat Edit feature,
 * the client should render codelens and decorations to improve the readability of the pending changes.
 * - method: `textDocument/codeLens`
 * - params: {@link CodeLensParams}
 * - result: {@link CodeLens}[] | null
 * - partialResult:  {@link CodeLens}[]
 */
declare namespace CodeLensRequest {
    const method: "textDocument/codeLens";
    const messageDirection: MessageDirection;
    const type: ProtocolRequestType<CodeLensParams, CodeLens[] | null, CodeLens[], void, void>;
}
type CodeLens = CodeLens$1 & {
    command?: ChatEditResolveCommand | Command;
    data?: {
        type: CodeLensType;
        line?: ChangesPreviewLineType;
    };
};
type CodeLensType = "previewChanges";
type ChangesPreviewLineType = "header" | "footer" | "commentsFirstLine" | "comments" | "waiting" | "inProgress" | "unchanged" | "inserted" | "deleted";
/**
 * Extends LSP method Completion Request(↩️)
 *
 * Note: Tabby provides this method capability *only* when the client has *NO* `textDocument/inlineCompletion` capability.
 * - method: `textDocument/completion`
 * - params: {@link CompletionParams}
 * - result: {@link CompletionList} | null
 */
declare namespace CompletionRequest {
    const method: "textDocument/completion";
    const messageDirection: MessageDirection;
    const type: ProtocolRequestType<CompletionParams, CompletionList | null, never, void, void>;
}
type CompletionList = CompletionList$1 & {
    items: CompletionItem[];
};
type CompletionItem = CompletionItem$1 & {
    data?: {
        /**
         * The eventId is for telemetry purposes, should be used in `tabby/telemetry/event`.
         */
        eventId?: CompletionEventId;
    };
};
type CompletionEventId = {
    completionId: string;
    choiceIndex: number;
};
/**
 * Extends LSP method Inline Completion Request(↩️)
 *
 * Note: Tabby provides this method capability only when the client has `textDocument/inlineCompletion` capability.
 * - method: `textDocument/inlineCompletion`
 * - params: {@link InlineCompletionParams}
 * - result: {@link InlineCompletionList} | null
 */
declare namespace InlineCompletionRequest {
    const method: "textDocument/inlineCompletion";
    const messageDirection: MessageDirection;
    const type: ProtocolRequestType<InlineCompletionParams, InlineCompletionList | null, never, void, void>;
}
type InlineCompletionList = InlineCompletionList$1 & {
    isIncomplete: boolean;
    items: InlineCompletionItem[];
};
type InlineCompletionItem = InlineCompletionItem$1 & {
    data?: {
        /**
         * The eventId is for telemetry purposes, should be used in `tabby/telemetry/event`.
         */
        eventId?: CompletionEventId;
    };
};
/**
 * [Tabby] Chat Edit Suggestion Command Request(↩️)
 *
 * This method is sent from the client to the server to get suggestion commands for the current context.
 * - method: `tabby/chat/edit/command`
 * - params: {@link ChatEditCommandParams}
 * - result: {@link ChatEditCommand}[] | null
 * - partialResult:  {@link ChatEditCommand}[]
 */
declare namespace ChatEditCommandRequest {
    const method = "tabby/chat/edit/command";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType<ChatEditCommandParams, ChatEditCommand[] | null, ChatEditCommand[], void, void>;
}
type ChatEditCommandParams = {
    /**
     * The document location to get suggestion commands for.
     */
    location: Location;
};
type ChatEditCommand = {
    /**
     * The display label of the command.
     */
    label: string;
    /**
     * A string value for the command.
     * If the command is a `preset` command, it always starts with `/`.
     */
    command: string;
    /**
     * The source of the command.
     */
    source: "preset";
};
/**
 * [Tabby] Chat Edit Request(↩️)
 *
 * This method is sent from the client to the server to edit the document content by user's command.
 * The server will edit the document content using ApplyEdit(`workspace/applyEdit`) request,
 * which requires the client to have this capability.
 * - method: `tabby/chat/edit`
 * - params: {@link ChatEditRequest}
 * - result: {@link ChatEditToken}
 * - error: {@link ChatFeatureNotAvailableError}
 *        | {@link ChatEditDocumentTooLongError}
 *        | {@link ChatEditCommandTooLongError}
 *        | {@link ChatEditMutexError}
 */
declare namespace ChatEditRequest {
    const method = "tabby/chat/edit";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType<ChatEditParams, string, void, ChatFeatureNotAvailableError | ChatEditDocumentTooLongError | ChatEditCommandTooLongError | ChatEditMutexError, void>;
}
type ChatEditParams = {
    /**
     * The document location to edit.
     */
    location: Location;
    /**
     * The command for this edit.
     * If the command is a `preset` command, it should always start with "/".
     * See {@link ChatEditCommand}
     */
    command: string;
    /**
     * Select a edit format.
     * - "previewChanges": The document will be edit to preview changes,
     *    use {@link ChatEditResolveRequest} to resolve it later.
     */
    format: "previewChanges";
};
type ChatEditToken = string;
type ChatFeatureNotAvailableError = {
    name: "ChatFeatureNotAvailableError";
};
type ChatEditDocumentTooLongError = {
    name: "ChatEditDocumentTooLongError";
};
type ChatEditCommandTooLongError = {
    name: "ChatEditCommandTooLongError";
};
type ChatEditMutexError = {
    name: "ChatEditMutexError";
};
/**
 * [Tabby] Chat Edit Resolve Request(↩️)
 *
 * This method is sent from the client to the server to accept or discard the changes in preview.
 * - method: `tabby/chat/edit/resolve`
 * - params: {@link ChatEditResolveParams}
 * - result: boolean
 */
declare namespace ChatEditResolveRequest {
    const method = "tabby/chat/edit/resolve";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType<ChatEditResolveParams, boolean, never, void, void>;
}
type ChatEditResolveParams = {
    /**
     * The document location to resolve the changes, should locate at the header line of the changes preview.
     */
    location: Location;
    /**
     * The action to take for this edit.
     */
    action: "accept" | "discard";
};
type ChatEditResolveCommand = Command & {
    title: string;
    tooltip?: string;
    command: "tabby/chat/edit/resolve";
    arguments: [ChatEditResolveParams];
};
/**
 * [Tabby] GenerateCommitMessage Request(↩️)
 *
 * This method is sent from the client to the server to generate a commit message for a git repository.
 * - method: `tabby/chat/generateCommitMessage`
 * - params: {@link GenerateCommitMessageParams}
 * - result: {@link GenerateCommitMessageResult} | null
 * - error: {@link ChatFeatureNotAvailableError}
 */
declare namespace GenerateCommitMessageRequest {
    const method = "tabby/chat/generateCommitMessage";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType<GenerateCommitMessageParams, GenerateCommitMessageResult | null, void, ChatFeatureNotAvailableError, void>;
}
type GenerateCommitMessageParams = {
    /**
     * The root URI of the git repository.
     */
    repository: URI;
};
type GenerateCommitMessageResult = {
    commitMessage: string;
};
/**
 * [Tabby] Telemetry Event Notification(➡️)
 *
 * This method is sent from the client to the server for telemetry purposes.
 * - method: `tabby/telemetry/event`
 * - params: {@link EventParams}
 * - result: void
 */
declare namespace TelemetryEventNotification {
    const method = "tabby/telemetry/event";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolNotificationType<EventParams, void>;
}
type EventParams = {
    type: "view" | "select" | "dismiss";
    selectKind?: "line";
    eventId: CompletionEventId;
    viewId?: string;
    elapsed?: number;
};
/**
 * [Tabby] DidUpdateServerInfo Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the current Tabby server info has changed.
 * - method: `tabby/agent/didUpdateServerInfo`
 * - params: {@link DidUpdateServerInfoParams}
 * - result: void
 */
declare namespace AgentServerInfoSync {
    const method = "tabby/agent/didUpdateServerInfo";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolNotificationType<DidUpdateServerInfoParams, void>;
}
type DidUpdateServerInfoParams = {
    serverInfo: ServerInfo;
};
type ServerInfo = {
    config: {
        endpoint: string;
        token: string | null;
        requestHeaders: Record<string, string | number | boolean | null | undefined> | null;
    };
    health: Record<string, unknown> | null;
};
/**
 * [Tabby] Server Info Request(↩️)
 *
 * This method is sent from the client to the server to check the current Tabby server info.
 * - method: `tabby/agent/serverInfo`
 * - params: none
 * - result: {@link ServerInfo}
 */
declare namespace AgentServerInfoRequest {
    const method = "tabby/agent/serverInfo";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType0<ServerInfo, never, void, void>;
}
/**
 * [Tabby] DidChangeStatus Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client about the status of the server.
 * - method: `tabby/agent/didChangeStatus`
 * - params: {@link DidChangeStatusParams}
 * - result: void
 */
declare namespace AgentStatusSync {
    const method = "tabby/agent/didChangeStatus";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolNotificationType<DidChangeStatusParams, void>;
}
type DidChangeStatusParams = {
    status: Status;
};
type Status = "notInitialized" | "ready" | "disconnected" | "unauthorized" | "finalized";
/**
 * [Tabby] Status Request(↩️)
 *
 * This method is sent from the client to the server to check the current status of the server.
 * - method: `tabby/agent/status`
 * - params: none
 * - result: {@link Status}
 */
declare namespace AgentStatusRequest {
    const method = "tabby/agent/status";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType0<Status, never, void, void>;
}
/**
 * [Tabby] DidUpdateIssue Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client about the current issues.
 * - method: `tabby/agent/didUpdateIssues`
 * - params: {@link DidUpdateIssueParams}
 * - result: void
 */
declare namespace AgentIssuesSync {
    const method = "tabby/agent/didUpdateIssues";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolNotificationType<IssueList, void>;
}
type DidUpdateIssueParams = IssueList;
type IssueList = {
    issues: IssueName[];
};
type IssueName = "slowCompletionResponseTime" | "highCompletionTimeoutRate" | "connectionFailed";
/**
 * [Tabby] Issues Request(↩️)
 *
 * This method is sent from the client to the server to check if there is any issue.
 * - method: `tabby/agent/issues`
 * - params: none
 * - result: {@link IssueList}
 */
declare namespace AgentIssuesRequest {
    const method = "tabby/agent/issues";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType0<IssueList, never, void, void>;
}
/**
 * [Tabby] Issue Detail Request(↩️)
 *
 * This method is sent from the client to the server to check the detail of an issue.
 * - method: `tabby/agent/issue/detail`
 * - params: {@link IssueDetailParams}
 * - result: {@link IssueDetailResult} | null
 */
declare namespace AgentIssueDetailRequest {
    const method = "tabby/agent/issue/detail";
    const messageDirection = MessageDirection.clientToServer;
    const type: ProtocolRequestType<IssueDetailParams, IssueDetailResult | null, never, void, void>;
}
type IssueDetailParams = {
    name: IssueName;
    helpMessageFormat?: "markdown" | "html";
};
type IssueDetailResult = {
    name: IssueName;
    helpMessage?: string;
};
/**
 * [Tabby] Read File Request(↪️)
 *
 * This method is sent from the server to the client to read the file contents.
 * - method: `tabby/workspaceFileSystem/readFile`
 * - params: {@link ReadFileParams}
 * - result: {@link ReadFileResult} | null
 */
declare namespace ReadFileRequest {
    const method = "tabby/workspaceFileSystem/readFile";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<ReadFileParams, ReadFileResult | null, never, void, void>;
}
type ReadFileParams = {
    uri: URI;
    /**
     * If `text` is select, the result should try to decode the file contents to string,
     * otherwise, the result should be a raw binary array.
     */
    format: "text";
    /**
     * When omitted, read the whole file.
     */
    range?: Range;
};
type ReadFileResult = {
    /**
     * If `text` is select, the result should be a string.
     */
    text?: string;
};
/**
 * [Tabby] DataStore Get Request(↪️)
 *
 * This method is sent from the server to the client to get the value of the given key.
 * - method: `tabby/dataStore/get`
 * - params: {@link DataStoreGetParams}
 * - result: any
 */
declare namespace DataStoreGetRequest {
    const method = "tabby/dataStore/get";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<DataStoreGetParams, any, never, void, void>;
}
type DataStoreGetParams = {
    key: string;
};
/**
 * [Tabby] DataStore Set Request(↪️)
 *
 * This method is sent from the server to the client to set the value of the given key.
 * - method: `tabby/dataStore/set`
 * - params: {@link DataStoreSetParams}
 * - result: boolean
 */
declare namespace DataStoreSetRequest {
    const method = "tabby/dataStore/set";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<DataStoreSetParams, boolean, never, void, void>;
}
type DataStoreSetParams = {
    key: string;
    value: any;
};
/**
 * [Tabby] Language Support Declaration Request(↪️)
 *
 * This method is sent from the server to the client to request the support from another language server.
 * See LSP `textDocument/declaration`.
 * - method: `tabby/languageSupport/textDocument/declaration`
 * - params: {@link DeclarationParams}
 * - result: {@link Declaration} | {@link LocationLink}[] | null
 */
declare namespace LanguageSupportDeclarationRequest {
    const method = "tabby/languageSupport/textDocument/declaration";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<DeclarationParams, Declaration | LocationLink[] | null, never, void, void>;
}
/**
 * [Tabby] Semantic Tokens Range Request(↪️)
 *
 * This method is sent from the server to the client to request the support from another language server.
 * See LSP `textDocument/semanticTokens/range`.
 * - method: `tabby/languageSupport/textDocument/semanticTokens/range`
 * - params: {@link SemanticTokensRangeParams}
 * - result: {@link SemanticTokensRangeResult} | null
 */
declare namespace LanguageSupportSemanticTokensRangeRequest {
    const method = "tabby/languageSupport/textDocument/semanticTokens/range";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<SemanticTokensRangeParams, SemanticTokensRangeResult | null, never, void, void>;
}
type SemanticTokensRangeResult = {
    legend: SemanticTokensLegend;
    tokens: SemanticTokens;
};
/**
 * [Tabby] Git Repository Request(↪️)
 *
 * This method is sent from the server to the client to get the git repository state of a file.
 * - method: `tabby/git/repository`
 * - params: {@link GitRepositoryParams}
 * - result: {@link GitRepository} | null
 */
declare namespace GitRepositoryRequest {
    const method = "tabby/git/repository";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<GitRepositoryParams, GitRepository | null, never, void, void>;
}
type GitRepositoryParams = {
    /**
     * The URI of the file to get the git repository state of.
     */
    uri: URI;
};
type GitRepository = {
    /**
     * The root URI of the git repository.
     */
    root: URI;
    /**
     * The url of the default remote.
     */
    remoteUrl?: string;
    /**
     * List of remotes in the git repository.
     */
    remotes?: {
        name: string;
        url: string;
    }[];
};
/**
 * [Tabby] Git Diff Request(↪️)
 *
 * This method is sent from the server to the client to get the diff of a git repository.
 * - method: `tabby/git/diff`
 * - params: {@link GitDiffParams}
 * - result: {@link GitDiffResult} | null
 */
declare namespace GitDiffRequest {
    const method = "tabby/git/diff";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<GitDiffParams, GitDiffResult | null, never, void, void>;
}
type GitDiffParams = {
    /**
     * The root URI of the git repository.
     */
    repository: URI;
    /**
     * Returns the cached or uncached diff of the git repository.
     */
    cached: boolean;
};
type GitDiffResult = {
    /**
     * The diff of the git repository.
     * - It could be the full diff.
     * - It could be a list of diff for each single file, sorted by the priority.
     *   This will be useful when the full diff is too large, and we will select
     *   from the split diffs to generate a prompt under the tokens limit.
     */
    diff: string | string[];
};
/**
 * [Tabby] Editor Options Request(↪️)
 *
 * This method is sent from the server to the client to get the diff of a git repository.
 * - method: `tabby/editorOptions`
 * - params: {@link EditorOptionsParams}
 * - result: {@link EditorOptions} | null
 */
declare namespace EditorOptionsRequest {
    const method = "tabby/editorOptions";
    const messageDirection = MessageDirection.serverToClient;
    const type: ProtocolRequestType<EditorOptionsParams, EditorOptions | null, never, void, void>;
}
type EditorOptionsParams = {
    /**
     * The uri of the document for which the editor options are requested.
     */
    uri: URI;
};
type EditorOptions = {
    /**
     * A string representing the indentation for the editor. It could be 2 or 4 spaces, or 1 tab.
     */
    indentation: string;
};

export { AgentIssueDetailRequest, AgentIssuesRequest, AgentIssuesSync, AgentServerInfoRequest, AgentServerInfoSync, AgentStatusRequest, AgentStatusSync, type ChangesPreviewLineType, type ChatEditCommand, type ChatEditCommandParams, ChatEditCommandRequest, type ChatEditCommandTooLongError, type ChatEditDocumentTooLongError, type ChatEditMutexError, type ChatEditParams, ChatEditRequest, type ChatEditResolveCommand, type ChatEditResolveParams, ChatEditResolveRequest, type ChatEditToken, type ChatFeatureNotAvailableError, ChatFeatureRegistration, type ClientCapabilities, type ClientInfo, type ClientProvidedConfig, type CodeLens, CodeLensRequest, type CodeLensType, type CompletionEventId, type CompletionItem, type CompletionList, CompletionRequest, ConfigurationRequest, type DataStoreGetParams, DataStoreGetRequest, type DataStoreSetParams, DataStoreSetRequest, DidChangeConfigurationNotification, type DidChangeConfigurationParams, type DidChangeStatusParams, type DidUpdateIssueParams, type DidUpdateServerInfoParams, type EditorOptions, type EditorOptionsParams, EditorOptionsRequest, type EventParams, type GenerateCommitMessageParams, GenerateCommitMessageRequest, type GenerateCommitMessageResult, type GitDiffParams, GitDiffRequest, type GitDiffResult, type GitRepository, type GitRepositoryParams, GitRepositoryRequest, type InitializeParams, InitializeRequest, type InitializeResult, type InlineCompletionItem, type InlineCompletionList, InlineCompletionRequest, type IssueDetailParams, type IssueDetailResult, type IssueList, type IssueName, LanguageSupportDeclarationRequest, LanguageSupportSemanticTokensRangeRequest, type ReadFileParams, ReadFileRequest, type ReadFileResult, type SemanticTokensRangeResult, type ServerCapabilities, type ServerInfo, type Status, TelemetryEventNotification };
