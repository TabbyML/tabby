/* eslint-disable @typescript-eslint/no-namespace */

import {
  ProtocolRequestType0,
  ProtocolRequestType,
  ProtocolNotificationType,
  RegistrationType,
  MessageDirection,
  LSPAny,
  URI,
  Range,
  Location,
  Command as LspCommand,
  InitializeRequest as LspInitializeRequest,
  InitializeParams as LspInitializeParams,
  InitializeResult as LspInitializeResult,
  InitializeError,
  ClientCapabilities as LspClientCapabilities,
  TextDocumentClientCapabilities,
  CompletionClientCapabilities,
  InlineCompletionClientCapabilities,
  ServerCapabilities as LspServerCapabilities,
  ConfigurationRequest as LspConfigurationRequest,
  DidChangeConfigurationNotification as LspDidChangeConfigurationNotification,
  DidChangeConfigurationParams as LspDidChangeConfigurationParams,
  CodeLensRequest as LspCodeLensRequest,
  CodeLensParams,
  CodeLens as LspCodeLens,
  CompletionRequest as LspCompletionRequest,
  CompletionParams,
  CompletionList as LspCompletionList,
  CompletionItem as LspCompletionItem,
  InlineCompletionRequest as LspInlineCompletionRequest,
  InlineCompletionParams,
  InlineCompletionList as LspInlineCompletionList,
  InlineCompletionItem as LspInlineCompletionItem,
  DeclarationParams,
  Declaration,
  LocationLink,
  SemanticTokensRangeParams,
  SemanticTokens,
  SemanticTokensLegend,
  WorkspaceEdit,
} from "vscode-languageserver-protocol";

/**
 * Extends LSP method Initialize Request(↩️)
 *
 * - method: `initialize`
 * - params: {@link InitializeParams}
 * - result: {@link InitializeResult}
 */
export namespace InitializeRequest {
  export const method = LspInitializeRequest.method;
  export const messageDirection = LspInitializeRequest.messageDirection;
  export const type = new ProtocolRequestType<InitializeParams, InitializeResult, InitializeError, void, void>(method);
}

export type InitializeParams = LspInitializeParams & {
  clientInfo?: ClientInfo;
  capabilities: ClientCapabilities;
  initializationOptions?: InitializationOptions;
};

export type InitializationOptions = {
  config?: ClientProvidedConfig;
  /**
   * ClientInfo also can be provided in InitializationOptions, will be merged with the one in InitializeParams.
   * This is useful for the clients that don't support changing the ClientInfo in InitializeParams.
   */
  clientInfo?: ClientInfo;
  /**
   * ClientCapabilities also can be provided in InitializationOptions, will be merged with the one in InitializeParams.
   * This is useful for the clients that don't support changing the ClientCapabilities in InitializeParams.
   */
  clientCapabilities?: ClientCapabilities;
  /**
   * The data store records that should be initialized when the server starts. This is useful for the clients that
   * provides the dataStore capability.
   */
  dataStoreRecords?: DataStoreRecords;
};

export type InitializeResult = LspInitializeResult & {
  capabilities: ServerCapabilities;
};

/**
 * [Tabby] Defines the name and version information of the IDE and the tabby plugin.
 */
export type ClientInfo = {
  name: string;
  version?: string;
  tabbyPlugin?: {
    name: string;
    version?: string;
  };
};

export type ClientCapabilities = LspClientCapabilities & {
  textDocument?: TextDocumentClientCapabilities & {
    completion?: boolean | CompletionClientCapabilities;
    inlineCompletion?: boolean | InlineCompletionClientCapabilities;
  };
  tabby?: {
    /**
     * @deprecated Use configListener and statusListener instead.
     * The client supports:
     * - `tabby/agent/didUpdateServerInfo`
     * - `tabby/agent/didChangeStatus`
     * - `tabby/agent/didUpdateIssues`
     * This capability indicates that client support receiving agent notifications.
     */
    agent?: boolean;
    /**
     * The client supports:
     * - `tabby/config/didChange`
     * This capability indicates that client support receiving notifications for configuration changes.
     */
    configDidChangeListener?: boolean;
    /**
     * The client supports:
     * - `tabby/status/didChange`
     * This capability indicates that client support receiving notifications for status sync to display a status bar.
     */
    statusDidChangeListener?: boolean;
    /**
     * The client supports:
     * - `tabby/workspaceFileSystem/readFile`
     * This capability improves the workspace code snippets context (RAG).
     * When not provided, the server will try to fallback to NodeJS provided `fs` module,
     *  which is not available in the browser.
     */
    workspaceFileSystem?: boolean;
    /**
     * The client provides a initial data store records for initialization and supports methods:
     * - `tabby/dataStore/didUpdate`
     * - `tabby/dataStore/update`
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

export type ServerCapabilities = LspServerCapabilities & {
  tabby?: {
    /**
     * @deprecated The feature availability will be updated by dynamic registration.
     * See {@link ChatFeatureRegistration}
     *
     * The server supports:
     * - `tabby/chat/edit`
     * - `tabby/chat/generateCommitMessage`
     * See {@link ChatFeatureRegistration}
     */
    chat?: boolean;
  };
};

export namespace ChatFeatures {
  export const type = new RegistrationType<void>("tabby/chat");
}

/**
 * Extends LSP method Configuration Request(↪️)
 *
 * - method: `workspace/configuration`
 * - params: any, not used
 * - result: [{@link ClientProvidedConfig}] (the array should contains only one ClientProvidedConfig item)
 */
export namespace ConfigurationRequest {
  export const method = LspConfigurationRequest.method;
  export const messageDirection = LspConfigurationRequest.messageDirection;
  export const type = new ProtocolRequestType<LSPAny, [ClientProvidedConfig], never, void, void>(method);
}

/**
 * [Tabby] Defines the config supported to be changed on the client side (IDE).
 */
export type ClientProvidedConfig = {
  /**
   * Specifies the endpoint and token for connecting to the Tabby server.
   */
  server?: {
    endpoint?: string;
    token?: string;
  };
  /**
   * Specifies the proxy for http/https requests.
   */
  proxy?: {
    url?: string;
    authorization?: string;
  };
  /**
   * Trigger mode should be implemented on the client side.
   * Sending this config to the server is for telemetry purposes.
   */
  inlineCompletion?: {
    triggerMode?: InlineCompletionTriggerMode;
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

export type InlineCompletionTriggerMode = "auto" | "manual";

/**
 * Extends LSP method DidChangeConfiguration Notification(➡️)
 * - method: `workspace/didChangeConfiguration`
 * - params: {@link DidChangeConfigurationParams}
 * - result: void
 */
export namespace DidChangeConfigurationNotification {
  export const method = LspDidChangeConfigurationNotification.method;
  export const messageDirection = LspDidChangeConfigurationNotification.messageDirection;
  export const type = new ProtocolNotificationType<DidChangeConfigurationParams, void>(method);
}

export type DidChangeConfigurationParams = LspDidChangeConfigurationParams & {
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
export namespace CodeLensRequest {
  export const method = LspCodeLensRequest.method;
  export const messageDirection = LspCodeLensRequest.messageDirection;
  export const type = new ProtocolRequestType<CodeLensParams, CodeLens[] | null, CodeLens[], void, void>(method);
}

export type CodeLens = LspCodeLens & {
  command?: ChatEditResolveCommand | LspCommand;
  data?: {
    type: CodeLensType;
    line?: ChangesPreviewLineType;
    text?: ChangesPreviewTextType;
  };
};

export type CodeLensType = "previewChanges";
export type ChangesPreviewLineType =
  | "header"
  | "footer"
  | "commentsFirstLine"
  | "comments"
  | "waiting"
  | "inProgress"
  | "unchanged"
  | "inserted"
  | "deleted";

export type ChangesPreviewTextType = "inserted" | "deleted";

/**
 * Extends LSP method Completion Request(↩️)
 *
 * Note: Tabby provides this method capability when the client has `textDocument/completion` capability.
 * - method: `textDocument/completion`
 * - params: {@link CompletionParams}
 * - result: {@link CompletionList} | null
 */
export namespace CompletionRequest {
  export const method = LspCompletionRequest.method;
  export const messageDirection = LspCompletionRequest.messageDirection;
  export const type = new ProtocolRequestType<CompletionParams, CompletionList | null, never, void, void>(method);
}

export type CompletionList = LspCompletionList & {
  items: CompletionItem[];
};

export type CompletionItem = LspCompletionItem & {
  data?: {
    /**
     * The eventId is for telemetry purposes, should be used in `tabby/telemetry/event`.
     */
    eventId?: CompletionEventId;
  };
};

export type CompletionEventId = {
  completionId: string;
  choiceIndex: number;
};

/**
 * Extends LSP method Inline Completion Request(↩️)
 *
 * Note: Tabby provides this method capability when the client has `textDocument/inlineCompletion` capability.
 * - method: `textDocument/inlineCompletion`
 * - params: {@link InlineCompletionParams}
 * - result: {@link InlineCompletionList} | null
 */
export namespace InlineCompletionRequest {
  export const method = LspInlineCompletionRequest.method;
  export const messageDirection = LspInlineCompletionRequest.messageDirection;
  export const type = new ProtocolRequestType<InlineCompletionParams, InlineCompletionList | null, never, void, void>(
    method,
  );
}

export type InlineCompletionList = LspInlineCompletionList & {
  isIncomplete: boolean;
  items: InlineCompletionItem[];
};

export type InlineCompletionItem = LspInlineCompletionItem & {
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
export namespace ChatEditCommandRequest {
  export const method = "tabby/chat/edit/command";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<
    ChatEditCommandParams,
    ChatEditCommand[] | null,
    ChatEditCommand[],
    void,
    void
  >(method);
}

export type ChatEditCommandParams = {
  /**
   * The document location to get suggestion commands for.
   */
  location: Location;
};

export type ChatEditCommand = {
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
export namespace ChatEditRequest {
  export const method = "tabby/chat/edit";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<
    ChatEditParams,
    ChatEditToken,
    void,
    ChatFeatureNotAvailableError | ChatEditDocumentTooLongError | ChatEditCommandTooLongError | ChatEditMutexError,
    void
  >(method);
}

export type ChatEditParams = {
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

  /**
   * list of file contexts.
   */
  context?: ChatEditFileContext[];
};

/**
 * Represents a file context use in {@link ChatEditParams}.
 */
export interface ChatEditFileContext {
  /**
   * The symbol in the user command that refer to this file context.
   */
  referrer: string;

  /**
   * The uri of the file.
   */
  uri: URI;

  /**
   * The context range in the file.
   * If the range is not provided, the whole file is considered.
   */
  range?: Range;
}

export type ChatEditToken = string;

export type ChatFeatureNotAvailableError = {
  name: "ChatFeatureNotAvailableError";
};
export type ChatEditDocumentTooLongError = {
  name: "ChatEditDocumentTooLongError";
};
export type ChatEditCommandTooLongError = {
  name: "ChatEditCommandTooLongError";
};
export type ChatEditMutexError = {
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
export namespace ChatEditResolveRequest {
  export const method = "tabby/chat/edit/resolve";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<ChatEditResolveParams, boolean, never, void, void>(method);
}

export type ChatEditResolveParams = {
  /**
   * The document location to resolve the changes, should locate at the header line of the changes preview.
   */
  location: Location;
  /**
   * The action to take for this edit.
   */
  action: "accept" | "discard" | "cancel";
};

/**
 * [Tabby] Apply workspace edit request(↪️)
 *
 * This method is sent from the server to client to apply edit in workspace with options.
 * - method: `tabby/workspace/applyEdit`
 * - params: {@link ApplyWorkspaceEditParams}
 * - result: boolean
 */
export namespace ApplyWorkspaceEditRequest {
  export const method = "tabby/workspace/applyEdit";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<ApplyWorkspaceEditParams, boolean, never, void, void>(method);
}

export interface ApplyWorkspaceEditParams {
  /**
   * An optional label of the workspace edit. This label is
   * presented in the user interface for example on an undo
   * stack to undo the workspace edit.
   */
  label?: string;
  /**
   * The edits to apply.
   */
  edit: WorkspaceEdit;
  options?: {
    /**
     * Add undo stop before making the edits.
     */
    readonly undoStopBefore: boolean;
    /**
     * Add undo stop after making the edits.
     */
    readonly undoStopAfter: boolean;
  };
}

export type ChatEditResolveCommand = LspCommand & {
  title: string;
  tooltip?: string;
  command: "tabby/chat/edit/resolve";
  arguments: [ChatEditResolveParams];
};

/**
 * [Tabby] Smart Apply Request(↩️)
 *
 * This method is sent from the client to the server to smart apply the text to the target location.
 * The server will edit the document content using ApplyEdit(`workspace/applyEdit`) request,
 * which requires the client to have this capability.
 * - method: `tabby/chat/smartApply`
 * - params: {@link SmartApplyParams}
 * - result: boolean
 * - error: {@link ChatFeatureNotAvailableError}
 *        | {@link ChatEditDocumentTooLongError}
 *        | {@link ChatEditMutexError}
 */
export namespace SmartApplyRequest {
  export const method = "tabby/chat/smartApply";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<
    SmartApplyParams,
    boolean,
    void,
    ChatFeatureNotAvailableError | ChatEditDocumentTooLongError | ChatEditMutexError,
    void
  >(method);
}

export type SmartApplyParams = {
  location: Location;
  text: string;
};

/**
 * [Tabby] Did Change Active Editor Notification(➡️)
 *
 * This method is sent from the client to server when the active editor changed.
 *
 *
 * - method: `tabby/editors/didChangeActiveEditor`
 * - params: {@link OpenedFileParams}
 * - result: void
 */
export namespace DidChangeActiveEditorNotification {
  export const method = "tabby/editors/didChangeActiveEditor";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolNotificationType<DidChangeActiveEditorParams, void>(method);
}
export type DidChangeActiveEditorParams = {
  activeEditor: Location;
  visibleEditors: Location[] | undefined;
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
export namespace GenerateCommitMessageRequest {
  export const method = "tabby/chat/generateCommitMessage";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<
    GenerateCommitMessageParams,
    GenerateCommitMessageResult | null,
    void,
    ChatFeatureNotAvailableError,
    void
  >(method);
}

export type GenerateCommitMessageParams = {
  /**
   * The root URI of the git repository.
   */
  repository: URI;
};

export type GenerateCommitMessageResult = {
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
export namespace TelemetryEventNotification {
  export const method = "tabby/telemetry/event";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolNotificationType<EventParams, void>(method);
}

export type EventParams = {
  type: "view" | "select" | "dismiss";
  selectKind?: "line";
  eventId: CompletionEventId;
  viewId?: string;
  elapsed?: number;
};

/**
 * @deprecated See {@link StatusDidChangeNotification} {@link ConfigDidChangeNotification}
 * [Tabby] DidUpdateServerInfo Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the current Tabby server info has changed.
 * - method: `tabby/agent/didUpdateServerInfo`
 * - params: {@link DidUpdateServerInfoParams}
 * - result: void
 */
export namespace AgentServerInfoSync {
  export const method = "tabby/agent/didUpdateServerInfo";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolNotificationType<DidUpdateServerInfoParams, void>(method);
}

/** @deprecated */
export type DidUpdateServerInfoParams = {
  serverInfo: ServerInfo;
};

/** @deprecated */
export type ServerInfo = {
  config: {
    endpoint: string;
    token: string | null;
    requestHeaders: Record<string, string | number | boolean | null | undefined> | null;
  };
  health: Record<string, unknown> | null;
};

/**
 * @deprecated See {@link StatusRequest} {@link ConfigRequest}
 * [Tabby] Server Info Request(↩️)
 *
 * This method is sent from the client to the server to check the current Tabby server info.
 * - method: `tabby/agent/serverInfo`
 * - params: none
 * - result: {@link ServerInfo}
 */
export namespace AgentServerInfoRequest {
  export const method = "tabby/agent/serverInfo";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType0<ServerInfo, never, void, void>(method);
}

/**
 * @deprecated See {@link StatusDidChangeNotification}
 * [Tabby] DidChangeStatus Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client about the status of the server.
 * - method: `tabby/agent/didChangeStatus`
 * - params: {@link DidChangeStatusParams}
 * - result: void
 */
export namespace AgentStatusSync {
  export const method = "tabby/agent/didChangeStatus";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolNotificationType<DidChangeStatusParams, void>(method);
}

/** @deprecated */
export type DidChangeStatusParams = {
  status: Status;
};

/** @deprecated */
export type Status = "notInitialized" | "ready" | "disconnected" | "unauthorized" | "finalized";

/**
 * @deprecated See {@link StatusRequest}
 * [Tabby] Status Request(↩️)
 *
 * This method is sent from the client to the server to check the current status of the server.
 * - method: `tabby/agent/status`
 * - params: none
 * - result: {@link Status}
 */
export namespace AgentStatusRequest {
  export const method = "tabby/agent/status";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType0<Status, never, void, void>(method);
}

/**
 * @deprecated See {@link StatusDidChangeNotification}
 * [Tabby] DidUpdateIssue Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client about the current issues.
 * - method: `tabby/agent/didUpdateIssues`
 * - params: {@link DidUpdateIssueParams}
 * - result: void
 */
export namespace AgentIssuesSync {
  export const method = "tabby/agent/didUpdateIssues";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolNotificationType<DidUpdateIssueParams, void>(method);
}

/** @deprecated */
export type DidUpdateIssueParams = IssueList;

/** @deprecated */
export type IssueList = {
  issues: IssueName[];
};

/** @deprecated */
export type IssueName = "slowCompletionResponseTime" | "highCompletionTimeoutRate" | "connectionFailed";

/**
 * @deprecated See {@link StatusRequest}
 * [Tabby] Issues Request(↩️)
 *
 * This method is sent from the client to the server to check if there is any issue.
 * - method: `tabby/agent/issues`
 * - params: none
 * - result: {@link IssueList}
 */
export namespace AgentIssuesRequest {
  export const method = "tabby/agent/issues";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType0<IssueList, never, void, void>(method);
}

/**
 * @deprecated See {@link StatusShowHelpMessageRequest}
 * [Tabby] Issue Detail Request(↩️)
 *
 * This method is sent from the client to the server to check the detail of an issue.
 * - method: `tabby/agent/issue/detail`
 * - params: {@link IssueDetailParams}
 * - result: {@link IssueDetailResult} | null
 */
export namespace AgentIssueDetailRequest {
  export const method = "tabby/agent/issue/detail";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<IssueDetailParams, IssueDetailResult | null, never, void, void>(method);
}

/** @deprecated */
export type IssueDetailParams = {
  name: IssueName;
  helpMessageFormat?: "plaintext" | "markdown" | "html";
};

/** @deprecated */
export type IssueDetailResult = {
  name: IssueName;
  helpMessage?: string;
};

/**
 * [Tabby] Config Request(↩️)
 *
 * This method is sent from the client to the server to get the current configuration.
 * - method: `tabby/config`
 * - params: any, not used
 * - result: {@link Config}
 */
export namespace ConfigRequest {
  export const method = "tabby/config";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<LSPAny, Config, never, void, void>(method);
}

export type Config = {
  server: {
    endpoint: string;
    token: string;
    requestHeaders: Record<string, string | number | boolean | null | undefined>;
  };
};

/**
 * [Tabby] Config DidChange Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client of the configuration changes.
 * - method: `tabby/config/didChange`
 * - params: {@link Config}
 * - result: void
 */
export namespace ConfigDidChangeNotification {
  export const method = "tabby/config/didChange";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolNotificationType<Config, void>(method);
}

/**
 * [Tabby] Status Request(↩️)
 *
 * This method is sent from the client to the server to check the current status of the server.
 * - method: `tabby/status`
 * - params: {@link StatusRequestParams}
 * - result: {@link StatusInfo}
 */
export namespace StatusRequest {
  export const method = "tabby/status";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<StatusRequestParams, StatusInfo, never, void, void>(method);
}

export type StatusRequestParams = {
  /**
   * Forces a recheck of the connection to the Tabby server, and waiting for result.
   */
  recheckConnection?: boolean;
};

/**
 * [Tabby] StatusInfo is used to display the status bar in the editor.
 */
export type StatusInfo = {
  status:
    | "connecting"
    | "unauthorized"
    | "disconnected"
    | "ready"
    | "readyForAutoTrigger"
    | "readyForManualTrigger"
    | "fetching"
    | "completionResponseSlow"
    | "rateLimitExceeded";
  tooltip?: string;
  /**
   * The health information of the server if available.
   */
  serverHealth?: Record<string, unknown>;
  /**
   * The action to take for this status.
   * - `disconnected` or `completionResponseSlow` -> StatusShowHelpMessageCommand
   * - others -> undefined
   */
  command?: StatusShowHelpMessageCommand | LspCommand;
  /**
   * The help message if available.
   * Only available when this status info is returned from {@link StatusRequest}, not provided in {@link StatusDidChangeNotification}.
   * Only available when the status is `disconnected` or `completionResponseSlow`.
   */
  helpMessage?: string;
};

/**
 * [Tabby] Status DidChange Notification(⬅️)
 *
 * This method is sent from the server to the client to notify the client of the status changes.
 * - method: `tabby/status/didChange`
 * - params: {@link StatusInfo}
 * - result: void
 */
export namespace StatusDidChangeNotification {
  export const method = "tabby/status/didChange";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolNotificationType<StatusInfo, void>(method);
}

/**
 * [Tabby] Status Show Help Message Request(↩️)
 *
 * This method is sent from the client to the server to request to show the help message for the current status.
 * The server will callback client to show request using ShowMessageRequest (`window/showMessageRequest`).
 * - method: `tabby/status/showHelpMessage`
 * - params: any, not used
 * - result: boolean
 */
export namespace StatusShowHelpMessageRequest {
  export const method = "tabby/status/showHelpMessage";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<LSPAny, boolean, never, void, void>(method);
}

export type StatusShowHelpMessageCommand = LspCommand & {
  title: string;
  command: "tabby/status/showHelpMessage";
  arguments: [LSPAny];
};

/**
 * [Tabby] Status Ignored Issues Edit Request(↩️)
 *
 * This method is sent from the client to the server to add or remove the issues that marked as ignored.
 * - method: `tabby/status/ignoredIssues/edit`
 * - params: {@link StatusIgnoredIssuesEditParams}
 * - result: boolean
 */
export namespace StatusIgnoredIssuesEditRequest {
  export const method = "tabby/status/ignoredIssues/edit";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolRequestType<StatusIgnoredIssuesEditParams, boolean, never, void, void>(method);
}

export type StatusIssuesName = "completionResponseSlow";

export type StatusIgnoredIssuesEditParams = {
  operation: "add" | "remove" | "removeAll";
  issues: StatusIssuesName | StatusIssuesName[];
};

/**
 * [Tabby] Read File Request(↪️)
 *
 * This method is sent from the server to the client to read the file contents.
 * - method: `tabby/workspaceFileSystem/readFile`
 * - params: {@link ReadFileParams}
 * - result: {@link ReadFileResult} | null
 */
export namespace ReadFileRequest {
  export const method = "tabby/workspaceFileSystem/readFile";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<ReadFileParams, ReadFileResult | null, never, void, void>(method);
}

export type ReadFileParams = {
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

export type ReadFileResult = {
  /**
   * If `text` is select, the result should be a string.
   */
  text?: string;
};

/**
 * @deprecated see {@link InitializationOptions} and {@link DataStoreDidUpdateNotification}
 * [Tabby] DataStore Get Request(↪️)
 *
 * This method is sent from the server to the client to get the value of the given key.
 * - method: `tabby/dataStore/get`
 * - params: {@link DataStoreGetParams}
 * - result: any
 */
export namespace DataStoreGetRequest {
  export const method = "tabby/dataStore/get";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<DataStoreGetParams, any, never, void, void>(method);
}

/** @deprecated */
export type DataStoreGetParams = {
  key: string;
};

/**
 * @deprecated see {@link DataStoreUpdateRequest}
 * [Tabby] DataStore Set Request(↪️)
 *
 * This method is sent from the server to the client to set the value of the given key.
 * - method: `tabby/dataStore/set`
 * - params: {@link DataStoreSetParams}
 * - result: boolean
 */
export namespace DataStoreSetRequest {
  export const method = "tabby/dataStore/set";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<DataStoreSetParams, boolean, never, void, void>(method);
}

/** @deprecated */
export type DataStoreSetParams = {
  key: string;
  value: any;
};

/**
 * [Tabby] DataStore DidUpdate Notification(➡️)
 *
 * This method is sent from the client to the server to notify that the data store records has been updated.
 * - method: `tabby/dataStore/didUpdate`
 * - params: {@link DataStoreDidChangeParams}
 */
export namespace DataStoreDidUpdateNotification {
  export const method = "tabby/dataStore/didUpdate";
  export const messageDirection = MessageDirection.clientToServer;
  export const type = new ProtocolNotificationType<DataStoreRecords, void>(method);
}

/**
 * [Tabby] DataStore Update Request(↪️)
 *
 * This method is sent from the server to the client to update the data store records.
 * - method: `tabby/dataStore/update`
 * - params: {@link DataStoreUpdateParams}
 * - result: boolean
 */
export namespace DataStoreUpdateRequest {
  export const method = "tabby/dataStore/update";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<DataStoreRecords, boolean, never, void, void>(method);
}

export type DataStoreRecords = Record<string, any>;

/**
 * [Tabby] Language Support Declaration Request(↪️)
 *
 * This method is sent from the server to the client to request the support from another language server.
 * See LSP `textDocument/declaration`.
 * - method: `tabby/languageSupport/textDocument/declaration`
 * - params: {@link DeclarationParams}
 * - result: {@link Declaration} | {@link LocationLink}[] | null
 */
export namespace LanguageSupportDeclarationRequest {
  export const method = "tabby/languageSupport/textDocument/declaration";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<
    DeclarationParams,
    Declaration | LocationLink[] | null,
    never,
    void,
    void
  >(method);
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
export namespace LanguageSupportSemanticTokensRangeRequest {
  export const method = "tabby/languageSupport/textDocument/semanticTokens/range";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<
    SemanticTokensRangeParams,
    SemanticTokensRangeResult | null,
    never,
    void,
    void
  >(method);
}

export type SemanticTokensRangeResult = {
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
export namespace GitRepositoryRequest {
  export const method = "tabby/git/repository";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<GitRepositoryParams, GitRepository | null, never, void, void>(method);
}

export type GitRepositoryParams = {
  /**
   * The URI of the file to get the git repository state of.
   */
  uri: URI;
};

export type GitRepository = {
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
export namespace GitDiffRequest {
  export const method = "tabby/git/diff";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<GitDiffParams, GitDiffResult | null, never, void, void>(method);
}

export type GitDiffParams = {
  /**
   * The root URI of the git repository.
   */
  repository: URI;
  /**
   * Returns the cached or uncached diff of the git repository.
   */
  cached: boolean;
};

export type GitDiffResult = {
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
export namespace EditorOptionsRequest {
  export const method = "tabby/editorOptions";
  export const messageDirection = MessageDirection.serverToClient;
  export const type = new ProtocolRequestType<EditorOptionsParams, EditorOptions | null, never, void, void>(method);
}

export type EditorOptionsParams = {
  /**
   * The uri of the document for which the editor options are requested.
   */
  uri: URI;
};

export type EditorOptions = {
  /**
   * A string representing the indentation for the editor. It could be 2 or 4 spaces, or 1 tab.
   */
  indentation: string;
};
