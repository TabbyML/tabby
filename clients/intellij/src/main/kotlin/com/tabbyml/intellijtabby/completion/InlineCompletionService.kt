package com.tabbyml.intellijtabby.completion

import com.intellij.openapi.Disposable
import com.intellij.openapi.application.invokeLater
import com.intellij.openapi.command.WriteCommandAction
import com.intellij.openapi.components.Service
import com.intellij.openapi.components.service
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Document
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.event.CaretEvent
import com.intellij.openapi.editor.event.DocumentEvent
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.fileEditor.FileEditorManagerEvent
import com.intellij.openapi.fileEditor.FileEditorManagerListener
import com.intellij.openapi.project.Project
import com.intellij.openapi.util.TextRange
import com.intellij.util.messages.Topic
import com.tabbyml.intellijtabby.events.CaretListener
import com.tabbyml.intellijtabby.events.DocumentListener
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.offsetInDocument
import com.tabbyml.intellijtabby.lsp.positionInDocument
import com.tabbyml.intellijtabby.lsp.protocol.EventParams
import com.tabbyml.intellijtabby.lsp.protocol.InlineCompletionParams
import com.tabbyml.intellijtabby.safeSyncPublisher
import com.tabbyml.intellijtabby.settings.SettingsService
import com.tabbyml.intellijtabby.settings.SettingsState
import kotlinx.coroutines.*
import kotlinx.coroutines.future.await
import org.eclipse.lsp4j.TextDocumentIdentifier

@Service(Service.Level.PROJECT)
class InlineCompletionService(private val project: Project) : Disposable {
  private val logger = Logger.getInstance(InlineCompletionService::class.java)
  private val messageBusConnection = project.messageBus.connect()
  private val editorManager = FileEditorManager.getInstance(project)
  private val scope = CoroutineScope(Dispatchers.IO)
  private val settings = service<SettingsService>()
  private val renderer = InlineCompletionRenderer()

  private suspend fun getServer() = project.serviceOrNull<ConnectionService>()?.getServerAsync()

  data class InlineCompletionContext(
    val request: Request,
    val job: Job,
    val response: Response? = null,
    val partialAccepted: Boolean = false,
  ) {
    data class Request(
      val editor: Editor,
      val document: Document,
      val modificationStamp: Long,
      val offset: Int,
      val manually: Boolean,
    ) {
      fun withManually(manually: Boolean = true): Request {
        return Request(editor, document, modificationStamp, offset, manually)
      }

      companion object {
        fun from(editor: Editor, offset: Int? = null): Request {
          val document = editor.document
          return Request(
            editor, document, document.modificationStamp, offset ?: editor.caretModel.primaryCaret.offset, false
          )
        }
      }
    }

    data class Response(
      val completionList: InlineCompletionList,
      val itemIndex: Int,
    )

    fun isMatch(editor: Editor, offset: Int?): Boolean {
      return this.request.editor == editor && this.request.document == editor.document && (this.partialAccepted || this.request.modificationStamp == editor.document.modificationStamp) && offset?.let { this.request.offset == it } != false
    }

    fun withResponse(completionList: InlineCompletionList?, itemIndex: Int = 0): InlineCompletionContext {
      return InlineCompletionContext(request, job, completionList?.let { Response(it, itemIndex) })
    }

    fun withUpdatedItemIndex(itemIndex: Int): InlineCompletionContext {
      val response = this.response ?: return this
      return withResponse(response.completionList, itemIndex)
    }

    fun withPartialAccepted(acceptedItem: InlineCompletionItem, acceptedLength: Int): InlineCompletionContext {
      val forwardRequest = Request(
        request.editor, request.document, request.modificationStamp, request.offset + acceptedLength, request.manually
      )
      val forwardResponse = Response(
        completionList = InlineCompletionList(
          true, listOf(
            InlineCompletionItem(
              insertText = acceptedItem.insertText, replaceRange = InlineCompletionItem.Range(
                acceptedItem.replaceRange.start, acceptedItem.replaceRange.end + acceptedLength
              ), data = acceptedItem.data
            )
          )
        ), itemIndex = 0
      )
      return InlineCompletionContext(forwardRequest, job, forwardResponse, true)
    }
  }

  private var current: InlineCompletionContext? = null
  private var currentContextWriteLock = Object()
  private var shouldAutoTrigger: Boolean
  private var documentChangedListenerJob: Job? = null

  fun isInlineCompletionVisibleAt(editor: Editor, offset: Int): Boolean =
    renderer.current?.editor == editor && renderer.current?.offset == offset

  fun isInlineCompletionStartWithIndentation(): Boolean {
    val renderingContext = renderer.current ?: return false
    val document = renderingContext.editor.document
    val completionItem = renderingContext.completionItem
    val offset = renderingContext.offset
    val lineStart = document.getLineStartOffset(document.getLineNumber(offset))
    val linePrefix = document.getText(TextRange(lineStart, offset))
    val visibleText = completionItem.insertText.substring(offset - completionItem.replaceRange.start)
    return linePrefix.isBlank() && visibleText.matches(Regex("(\\t+| {2,}).*", RegexOption.DOT_MATCHES_ALL))
  }

  init {
    logger.debug("Initialize InlineCompletionService.")
    val triggerMode = settings.settings().completionTriggerMode
    logger.debug("TriggerMode: $triggerMode")
    shouldAutoTrigger = triggerMode == SettingsState.TriggerMode.AUTOMATIC
    messageBusConnection.subscribe(SettingsService.Listener.TOPIC, object : SettingsService.Listener {
      override fun settingsChanged(settings: SettingsService.Settings) {
        logger.debug("TriggerMode updated: ${settings.completionTriggerMode}")
        shouldAutoTrigger = triggerMode == SettingsState.TriggerMode.AUTOMATIC
      }
    })
    messageBusConnection.subscribe(DocumentListener.TOPIC, object : DocumentListener {
      override fun documentChanged(document: Document, editor: Editor, event: DocumentEvent) {
        if (editorManager.selectedTextEditor == editor) {
          documentChangedListenerJob?.cancel()
          documentChangedListenerJob = scope.launch {
            // debounce for multiple documentChanged events triggered at one stroke
            // such as input `(`, and `)` will be auto added, this will trigger two events
            delay(10)
            handleDocumentChanged(document, editor, event)
          }
        }
      }
    })
    messageBusConnection.subscribe(CaretListener.TOPIC, object : CaretListener {
      override fun caretPositionChanged(editor: Editor, event: CaretEvent) {
        if (editorManager.selectedTextEditor == editor) {
          val offset = editor.caretModel.offset
          val context = current
          if (context != null && !context.isMatch(editor, offset)) {
            dismiss()
          }
        }
      }
    })
    messageBusConnection.subscribe(FileEditorManagerListener.FILE_EDITOR_MANAGER, object : FileEditorManagerListener {
      override fun selectionChanged(event: FileEditorManagerEvent) {
        dismiss()
      }
    })
  }

  private fun handleDocumentChanged(document: Document, editor: Editor, event: DocumentEvent) {
    if (event.newFragment.isEmpty()) {
      // newFragment is empty, so this is a delete or backspace, which do not trigger caret position changed,
      // so we should handle it here
      val offset = event.offset
      val context = current
      if (context != null && !context.isMatch(editor, offset)) {
        dismiss()
      }
    }
    if (shouldAutoTrigger) {
      invokeLater {
        val offset = editor.caretModel.offset
        provideInlineCompletion(editor, offset)
      }
    }
  }

  private fun buildInlineCompletionParams(requestContext: InlineCompletionContext.Request): InlineCompletionParams {
    val documentUri = requestContext.editor.virtualFile.url
    return InlineCompletionParams(
      context = InlineCompletionParams.InlineCompletionContext(
        triggerKind = if (requestContext.manually) {
          InlineCompletionParams.InlineCompletionContext.InlineCompletionTriggerKind.Invoked
        } else {
          InlineCompletionParams.InlineCompletionContext.InlineCompletionTriggerKind.Automatic
        }
      ),
      textDocument = TextDocumentIdentifier(documentUri),
      position = positionInDocument(requestContext.document, requestContext.offset),
    )
  }

  private fun convertInlineCompletionList(
    inlineCompletionList: com.tabbyml.intellijtabby.lsp.protocol.InlineCompletionList?,
    requestContext: InlineCompletionContext.Request
  ): InlineCompletionList? {
    return inlineCompletionList?.let { list ->
      InlineCompletionList(
        isIncomplete = list.isIncomplete,
        items = list.items.map { item ->
          InlineCompletionItem(insertText = item.insertText, replaceRange = item.range?.let {
            InlineCompletionItem.Range(
              start = offsetInDocument(requestContext.document, it.start),
              end = offsetInDocument(requestContext.document, it.end),
            )
          } ?: InlineCompletionItem.Range(
            start = requestContext.offset,
            end = requestContext.offset,
          ), data = InlineCompletionItem.Data(
            eventId = item.data?.eventId
          ))
        },
      )
    }
  }

  private fun launchJobForInlineCompletion(
    requestContext: InlineCompletionContext.Request,
    finishedCallback: (inlineCompletionList: com.tabbyml.intellijtabby.lsp.protocol.InlineCompletionList?) -> Unit = {}
  ): Job {
    return scope.launch {
      val params = buildInlineCompletionParams(requestContext)
      val server = getServer() ?: return@launch
      logger.debug("Request inline completion: $params")
      project.safeSyncPublisher(Listener.TOPIC)?.loadingStateChanged(true)
      val inlineCompletionList = server.textDocumentFeature.inlineCompletion(params).await()
      val context = current
      if (requestContext == context?.request) {
        logger.debug("Request inline completion done: $inlineCompletionList")
        project.safeSyncPublisher(Listener.TOPIC)?.loadingStateChanged(false)
        finishedCallback(inlineCompletionList)
      } else if (context == null) {
        project.safeSyncPublisher(Listener.TOPIC)?.loadingStateChanged(false)
      }
    }
  }

  private fun renderCurrentResponse() {
    val context = current ?: return
    val editor = context.request.editor
    val offset = context.request.offset
    val completionItem = context.response?.let {
      it.completionList.items.getOrNull(it.itemIndex)
    }
    if (completionItem == null) {
      renderer.hide()
    } else {
      renderer.show(editor, offset, completionItem) {
        telemetryEvent(EventParams.EventType.VIEW, it)
      }
    }
  }

  private fun calcCycleIndex(index: Int, size: Int, direction: CycleDirection): Int {
    if (size <= 1) {
      return index
    }
    return when (direction) {
      CycleDirection.NEXT -> (index + 1).mod(size)
      CycleDirection.PREVIOUS -> (index - 1).mod(size)
    }
  }

  private fun telemetryEvent(
    type: String, renderingContext: InlineCompletionRenderer.RenderingContext, acceptType: AcceptType? = null
  ) {
    scope.launch {
      getServer()?.telemetryFeature?.event(
        EventParams(
          type = type,
          selectKind = when (acceptType) {
            AcceptType.NEXT_WORD, AcceptType.NEXT_LINE -> EventParams.SelectKind.LINE
            else -> null
          },
          eventId = renderingContext.completionItem.data?.eventId,
          viewId = renderingContext.id,
          elapsed = when (type) {
            EventParams.EventType.VIEW -> null
            else -> renderingContext.calcElapsed().toInt()
          },
        )
      )
    }
  }

  fun provideInlineCompletion(editor: Editor, offset: Int, manually: Boolean = false) {
    synchronized(currentContextWriteLock) {
      logger.debug("Provide inline completion at $editor $offset $manually")
      var partialAcceptedResponse: InlineCompletionContext.Response? = null
      current?.let {
        it.job.cancel()
        if (it.partialAccepted) {
          partialAcceptedResponse = it.response
        } else {
          renderer.hide()
        }
        current = null
      }
      val requestContext = InlineCompletionContext.Request.from(editor, offset).withManually(manually)
      val job = launchJobForInlineCompletion(requestContext) { inlineCompletionList ->
        synchronized(currentContextWriteLock) {
          current = current?.withResponse(convertInlineCompletionList(inlineCompletionList, requestContext))
          renderCurrentResponse()
        }
      }
      current = InlineCompletionContext(requestContext, job, partialAcceptedResponse)
    }
  }

  enum class CycleDirection {
    NEXT, PREVIOUS,
  }

  fun cycle(editor: Editor, offset: Int?, direction: CycleDirection) {
    synchronized(currentContextWriteLock) {
      logger.debug("Cycle inline completion at $editor $offset $direction")
      val context = current ?: return
      if (!context.isMatch(editor, offset)) {
        return
      }
      val responseContext = context.response ?: return
      val itemIndex = responseContext.itemIndex
      if (responseContext.completionList.isIncomplete) {
        val requestContext = context.request.withManually()
        val job = launchJobForInlineCompletion(requestContext) { inlineCompletionList ->
          inlineCompletionList?.let {
            synchronized(currentContextWriteLock) {
              current = current?.withResponse(convertInlineCompletionList(inlineCompletionList, requestContext))
                ?.withUpdatedItemIndex(calcCycleIndex(itemIndex, it.items.size, direction))
              renderCurrentResponse()
            }
          }
        }
        current = InlineCompletionContext(requestContext, job, responseContext)
      } else {
        current =
          current?.withUpdatedItemIndex(calcCycleIndex(itemIndex, responseContext.completionList.items.size, direction))
        renderCurrentResponse()
      }
    }
  }

  enum class AcceptType {
    FULL_COMPLETION, NEXT_WORD, NEXT_LINE,
  }

  fun accept(editor: Editor, offset: Int?, type: AcceptType) {
    synchronized(currentContextWriteLock) {
      logger.debug("Accept inline completion at $editor $offset $type")
      val context = current ?: return
      if (!context.isMatch(editor, offset)) {
        return
      }
      val originOffset = context.request.offset
      val completionItem = context.response?.let {
        it.completionList.items.getOrNull(it.itemIndex)
      }
      if (completionItem == null) {
        return
      }
      val prefixReplaceLength = originOffset - completionItem.replaceRange.start
      val completionText = completionItem.insertText.substring(prefixReplaceLength)
      val text = when (type) {
        AcceptType.FULL_COMPLETION -> completionText
        AcceptType.NEXT_WORD -> {
          Regex("\\w+|\\W+").find(completionText)?.value ?: ""
        }

        AcceptType.NEXT_LINE -> {
          val lines = completionText.lines()
          if (lines.size <= 1) {
            completionText
          } else if (lines.first().isEmpty()) {
            lines.subList(0, 2).joinToString("\n")
          } else {
            lines.first()
          }
        }
      }
      renderer.current?.let {
        telemetryEvent(EventParams.EventType.SELECT, it, type)
      }
      if (type == AcceptType.FULL_COMPLETION) {
        current = null
        renderer.hide()
      } else {
        current = context.withPartialAccepted(completionItem, text.length)
      }
      invokeLater {
        WriteCommandAction.runWriteCommandAction(editor.project) {
          editor.document.replaceString(originOffset, completionItem.replaceRange.end, text)
          editor.caretModel.moveToOffset(originOffset + text.length)
        }
        renderCurrentResponse()
      }
    }
  }

  fun dismiss() {
    synchronized(currentContextWriteLock) {
      logger.debug("Dismiss inline completion.")
      renderer.current?.let {
        telemetryEvent(EventParams.EventType.DISMISS, it)
        renderer.hide()
      }
      current?.let {
        it.job.cancel()
        project.safeSyncPublisher(Listener.TOPIC)?.loadingStateChanged(false)
        current = null
      }
    }
  }

  override fun dispose() {
    dismiss()
    messageBusConnection.dispose()
  }

  interface Listener {
    fun loadingStateChanged(loading: Boolean) {}

    companion object {
      @Topic.ProjectLevel
      val TOPIC = Topic(Listener::class.java, Topic.BroadcastDirection.NONE)
    }
  }
}
