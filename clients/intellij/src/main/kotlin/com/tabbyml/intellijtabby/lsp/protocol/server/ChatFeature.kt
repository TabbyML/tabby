package com.tabbyml.intellijtabby.lsp.protocol.server

import com.jetbrains.rd.generator.nova.PredefinedType
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditParams
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditResolveParams
import com.tabbyml.intellijtabby.lsp.protocol.GenerateCommitMessageParams
import com.tabbyml.intellijtabby.lsp.protocol.GenerateCommitMessageResult
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("tabby/chat")
interface ChatFeature {
  @JsonRequest("edit")
  fun chatEdit(params: ChatEditParams): CompletableFuture<String>

  @JsonRequest("edit/resolve")
  fun resolveEdit(params: ChatEditResolveParams): CompletableFuture<Boolean>

  @JsonRequest
  fun generateCommitMessage(params: GenerateCommitMessageParams): CompletableFuture<GenerateCommitMessageResult>
}
