package com.tabbyml.intellijtabby.lsp.protocol.server

import com.jetbrains.rd.generator.nova.PredefinedType
import com.tabbyml.intellijtabby.lsp.protocol.*
import org.eclipse.lsp4j.jsonrpc.services.JsonRequest
import org.eclipse.lsp4j.jsonrpc.services.JsonSegment
import java.util.concurrent.CompletableFuture

@JsonSegment("tabby/chat")
interface ChatFeature {
  @JsonRequest("edit")
  fun chatEdit(params: ChatEditParams): CompletableFuture<String>

  @JsonRequest("edit/resolve")
  fun resolveEdit(params: ChatEditResolveParams): CompletableFuture<Boolean>


  @JsonRequest("edit/command")
  fun editCommand(params: ChatEditCommandParams): CompletableFuture<List<ChatEditCommand>?>

  @JsonRequest
  fun generateCommitMessage(params: GenerateCommitMessageParams): CompletableFuture<GenerateCommitMessageResult>
}
