package com.tabbyml.intellijtabby.lsp.protocol.server

import com.tabbyml.intellijtabby.lsp.protocol.GenerateCommitMessageParams
import com.tabbyml.intellijtabby.lsp.protocol.GenerateCommitMessageResult
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("tabby/chat")
interface ChatFeature {
  @JsonRequest
  fun generateCommitMessage(params: GenerateCommitMessageParams): CompletableFuture<GenerateCommitMessageResult>
}
