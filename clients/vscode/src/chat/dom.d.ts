// We don't include DOM types in typescript building config

// FIXME: This is required by `@quilted/threads` and `tabby-chat-panel`
declare type HTMLIFrameElement = unknown;

declare type KeyboardEventInit = unknown;
