package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.intention.impl.BaseIntentionAction
import com.intellij.codeInsight.intention.preview.IntentionPreviewInfo
import com.intellij.icons.AllIcons
import com.intellij.ide.ui.LafManagerListener
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.diagnostic.Logger
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.EditorCustomElementRenderer
import com.intellij.openapi.editor.Inlay
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.EditorColorsScheme
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.intellij.psi.PsiFile
import com.intellij.ui.components.IconLabelButton
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.UIUtil
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.Range
import java.awt.*
import java.awt.event.FocusAdapter
import java.awt.event.FocusEvent
import java.awt.event.KeyEvent
import java.awt.event.KeyListener
import javax.swing.BorderFactory
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JTextArea
import com.intellij.codeInsight.codeVision.settings.CodeVisionSettings
import com.intellij.codeInsight.hints.settings.InlaySettingsConfigurable
import com.intellij.openapi.options.ShowSettingsUtil
import com.tabbyml.intellijtabby.actions.chat.isChatFeatureEnabled

class InlineChatIntentionAction : BaseIntentionAction(), DumbAware {
    private var inlay: Inlay<InlineChatInlayRenderer>? = null
    private var inlayRender: InlineChatInlayRenderer? = null
    private var project: Project? = null
    private var editor: Editor? = null
    override fun getFamilyName(): String {
        return "Tabby"
    }

    override fun isAvailable(project: Project, editor: Editor?, file: PsiFile?): Boolean {
        return editor != null && isChatFeatureEnabled(project);
    }

    override fun invoke(project: Project, editor: Editor?, file: PsiFile?) {
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return
        if (inlineChatService.inlineChatInputVisible || inlineChatService.hasDiffAction) return
        this.project = project
        this.editor = editor
        if (editor != null) {
            val locationInfo = getCurrentLocation(editor = editor)
            inlineChatService.location = locationInfo.location
            inlineChatService.inlineChatInputVisible = true
            addInputToEditor(project, editor, locationInfo.startOffset);
        }

        // listen for theme change
        project.messageBus.connect().subscribe(LafManagerListener.TOPIC, LafManagerListener {
            inlayRender?.repaint() // FIXME
        })
    }

    override fun getText(): String {
        return "Open Tabby inline edit";
    }

    override fun generatePreview(project: Project, editor: Editor, file: PsiFile): IntentionPreviewInfo {
        return IntentionPreviewInfo.EMPTY
    }

    private fun addInputToEditor(project: Project, editor: Editor, offset: Int) {
        val inlayModel = editor.inlayModel
        inlayRender = InlineChatInlayRenderer(project, editor, this::onClose, this::onInputSubmit)
        inlay = inlayModel.addBlockElement(offset, true, true, 0, inlayRender!!)
    }

    private fun onClose() {
        inlay?.dispose()
        val inlineChatService = project?.serviceOrNull<InlineChatService>() ?: return
        inlineChatService.inlineChatInputVisible = false
    }

    private fun onInputSubmit(value: String) {
        chatEdit(command = value)
        editor?.selectionModel?.removeSelection()
        project?.serviceOrNull<CommandHistory>()?.addCommand(value)
    }

    private fun chatEdit(command: String) {
        val scope = CoroutineScope(Dispatchers.IO)
        val inlineChatService = project?.serviceOrNull<InlineChatService>() ?: return
        scope.launch {
            val server = project?.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch
            val location = inlineChatService.location ?: return@launch
            val param = ChatEditParams(
                location = location,
                command = command
            )
            server.chatFeature.chatEdit(params = param)
        }
    }
}

class InlineChatInlayRenderer(
    private val project: Project,
    private val editor: Editor,
    private val onClose: () -> Unit,
    private val onSubmit: (value: String) -> Unit
) :
    EditorCustomElementRenderer {
    private val inlineChatComponent = InlineChatComponent(project, this::handleClose, onSubmit)
    private var targetRegion: Rectangle? = null
    private var disposed = false

    private val keyEventHandler = object : KeyEventDispatcher {
        override fun dispatchKeyEvent(e: KeyEvent): Boolean {
            if (e.id == KeyEvent.KEY_PRESSED && (e.keyCode == KeyEvent.VK_BACK_SPACE || e.keyCode == KeyEvent.VK_DELETE)) {
                if (e.component is JBTextArea) {
                    // Return true to consume the event (prevent default handling)
                    return true
                }
            }
            if (e.id == KeyEvent.KEY_PRESSED && e.keyCode == KeyEvent.VK_ESCAPE) {
                handleClose()
                return true
            }
            return false
        }
    }

    init {
        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(keyEventHandler)
    }


    override fun calcWidthInPixels(inlay: Inlay<*>): Int {
        return inlineChatComponent.preferredSize.width
    }

    override fun calcHeightInPixels(inlay: Inlay<*>): Int {
        return inlineChatComponent.preferredSize.height
    }

    override fun paint(inlay: Inlay<*>, g: Graphics, targetRegion: Rectangle, textAttributes: TextAttributes) {
        if (disposed) {
            return
        }
        val visibleArea = editor.scrollingModel.visibleArea
        if (this.targetRegion == null) {
            this.targetRegion = targetRegion
            this.targetRegion?.y = targetRegion.y + visibleArea.y
        }
        val firstTargetRegion = this.targetRegion ?: targetRegion
        inlineChatComponent.setSize(firstTargetRegion.width, firstTargetRegion.height)
        inlineChatComponent.setLocation(firstTargetRegion.x, firstTargetRegion.y)

        if (inlineChatComponent.parent == null) {
            editor.contentComponent.add(inlineChatComponent)
            inlineChatComponent.requestFocus()
        }
    }

    fun repaint() {
        inlineChatComponent.repaint()
    }

    private fun handleClose() {
        this.dispose()
        onClose()
    }


    private fun dispose() {
        if (disposed) {
            return
        }
        KeyboardFocusManager.getCurrentKeyboardFocusManager().removeKeyEventDispatcher(keyEventHandler)
        inlineChatComponent.parent?.remove(inlineChatComponent)
        editor.contentComponent.remove(inlineChatComponent)
        editor.contentComponent.repaint();
        this.disposed = true
    }
}

class InlineChatComponent(
    private val project: Project,
    private val onClose: () -> Unit,
    private val onSubmit: (value: String) -> Unit
) : JPanel() {
    private val closeButton = createCloseButton()
    private val inlineInput = InlineInputComponent(project, this::handleSubmit, this::handleClose)

    override fun isOpaque(): Boolean {
        return false;
    }

    private fun getTheme(): EditorColorsScheme {
        return EditorColorsManager.getInstance().globalScheme;
    }

    init {
        layout = BorderLayout()
        putClientProperty(UIUtil.HIDE_EDITOR_FROM_DATA_CONTEXT_PROPERTY, true)

        val contentPanel = JPanel(BorderLayout())
        contentPanel.add(inlineInput, BorderLayout.CENTER)
        contentPanel.add(closeButton, BorderLayout.EAST)
        contentPanel.border = BorderFactory.createEmptyBorder(6, 6, 6, 6)

        add(contentPanel, BorderLayout.CENTER)

        border = BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(getTheme().defaultBackground, 3, true),
            BorderFactory.createEmptyBorder(4, 8, 4, 8)
        )

        minimumSize = Dimension(200, 40)
        preferredSize = Dimension(800, 60)
    }

    private fun handleClose(): Unit {
        onClose()
    }

    private fun createCloseButton(): JLabel {
        val closeButton = IconLabelButton(AllIcons.Actions.Close) { handleClose() }
        closeButton.toolTipText = "Close Tabby inline edit"
        closeButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        closeButton.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        return closeButton
    }

    private fun handleSubmit(value: String) {
        onClose()
        onSubmit(value)
    }

    override fun requestFocus() {
        inlineInput.requestFocus()
    }
}

data class ChatEditFileContext(val referrer: String, val uri: String, val range: Range)
data class InlineEditCommand(val command: String, val context: List<ChatEditFileContext>?)

class InlineInputComponent(
    private var project: Project,
    private var onSubmit: (value: String) -> Unit,
    private var onCancel: () -> Unit
) : JPanel() {
    private val logger = Logger.getInstance(InlineInputComponent::class.java)
    private val history: CommandHistory? = project.serviceOrNull<CommandHistory>()
    private val textArea: JTextArea = createTextArea()
    private val submitButton: JLabel = createSubmitButton()
    private val historyButton: JLabel = createHistoryButton()
    private var commandListComponent: CommandListComponent? = null

    init {
        putClientProperty(UIUtil.HIDE_EDITOR_FROM_DATA_CONTEXT_PROPERTY, true)
        layout = BorderLayout()
        add(historyButton, BorderLayout.WEST)
        add(textArea, BorderLayout.CENTER)
        add(submitButton, BorderLayout.EAST)
        border = BorderFactory.createLineBorder(UIUtil.getHeaderInactiveColor(), 2, true)

        addKeyListener(object : KeyListener {
            override fun keyPressed(e: KeyEvent) {
                e.consume()
            }

            override fun keyReleased(e: KeyEvent) {
                if (e.keyCode == KeyEvent.VK_BACK_SPACE || e.keyCode == KeyEvent.VK_DELETE) {
                    e.consume()
                }
            }

            override fun keyTyped(e: KeyEvent) {
                e.consume()
            }
        })
    }

    private fun createTextArea(): JBTextArea {
        val textArea = JBTextArea().apply {
            font = Font(font.family, font.style, 14)
        }
        textArea.lineWrap = false
        textArea.rows = 1
        textArea.columns = 30
        textArea.emptyText.text = "Enter the command to editing"
        textArea.border = BorderFactory.createEmptyBorder(6, 4, 4, 4)
        // To prevent keystrokes(backspace, delete) being handled by the host editor, https://intellij-support.jetbrains.com/hc/en-us/community/posts/360010505760-Issues-embedding-editor-in-block-inlay
        textArea.putClientProperty(UIUtil.HIDE_EDITOR_FROM_DATA_CONTEXT_PROPERTY, true)
        textArea.addKeyListener(object : KeyListener {
            override fun keyPressed(e: KeyEvent) {
                //
            }

            override fun keyReleased(e: KeyEvent) {
                if (e.keyCode == KeyEvent.VK_BACK_SPACE || e.keyCode == KeyEvent.VK_DELETE) {
                    e.consume()
                }

                if (e.keyCode == KeyEvent.VK_ENTER) {
                    handleConfirm()
                }

                if (e.keyCode == KeyEvent.VK_ESCAPE) {
                    // Handle escape
                    textArea.text = ""
                    onCancel()
                }
            }

            override fun keyTyped(e: KeyEvent) {
                //
            }
        })
        textArea.addFocusListener(object : FocusAdapter() {
            override fun focusGained(e: FocusEvent) {
                border = BorderFactory.createLineBorder(UIUtil.getFocusedBorderColor(), 2, true)
            }

            override fun focusLost(e: FocusEvent) {
                border = BorderFactory.createLineBorder(UIUtil.getHeaderInactiveColor(), 2, true)
            }
        })
        return textArea
    }

    override fun requestFocus() {
        textArea.requestFocus()
    }

    private fun handleConfirm() {
        val codeVisionSettings = CodeVisionSettings.instance()
        if (!codeVisionSettings.isProviderEnabled("Tabby.InlineEdit")) {
            val result = Messages.showOkCancelDialog(
                project,
                "Tabby Inline Edit Code Vision feature is not enabled. Please enable it in Settings > Editor > Inlay Hint > Code Vision > Tabby Inline Edit.",
                "Tabby Inline Edit Code Vision Disabled",
                "Open Settings",
                "Cancel",
                Messages.getWarningIcon()
            )
            if (result == Messages.OK) {
                ShowSettingsUtil.getInstance().showSettingsDialog(project, InlaySettingsConfigurable::class.java)
            }
            textArea.text = textArea.text.trim()
            return
        }
        onSubmit(textArea.text.trim())
        textArea.text = ""
    }

    private fun createSubmitButton(): JLabel {
        val submitButton = IconLabelButton(AllIcons.Chooser.Right) { handleConfirm() }
        submitButton.toolTipText = "Submit the command"
        submitButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        submitButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)
        return submitButton
    }

    private fun createHistoryButton(): JLabel {
        val historyButton = IconLabelButton(AllIcons.Actions.SearchWithHistory) { handleOpenHistory() }
        historyButton.toolTipText = "Select suggested / history Command"
        historyButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        historyButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)
        return historyButton
    }

    private fun handleOpenHistory() {
        val commandItems = getCommandList()
        var popup: com.intellij.openapi.ui.popup.JBPopup? = null
        commandListComponent = CommandListComponent("Select Command", commandItems, {
            textArea.text = it.value
            popup?.cancel()
        }, {
            history?.deleteCommand(it.value)
            refreshCommandList() {
                popup?.pack(true, true)
            }
        }, {
            history?.clearHistory()
            refreshCommandList() {
                popup?.pack(true, true)
            }
        })
        popup =
            JBPopupFactory.getInstance().createComponentPopupBuilder(commandListComponent?.component!!, JPanel())
                .createPopup()
        popup.showUnderneathOf(this)
    }

    private fun refreshCommandList(onUpdated: (() -> Unit)? = null) {
        val commandItems = getCommandList()
        commandListComponent?.setData(commandItems, onUpdated)
    }

    private fun getHistoryCommand(): List<InlineEditCommand> {
        return history?.getHistory()?.map {
            InlineEditCommand(it, null)
        } ?: emptyList()
    }

    private fun getCommandList(): List<CommandListItem> {
        val location = project.serviceOrNull<InlineChatService>()?.location ?: return emptyList()
        val suggestedItems = try {
            getSuggestedCommands(project, location).get()?.map {
                CommandListItem(
                    label = it.label,
                    value = it.command,
                    icon = AllIcons.Actions.IntentionBulbGrey,
                    description = it.command,
                    canDelete = false
                )
            } ?: emptyList()
        } catch (e: Exception) {
            logger.warn("Error getting suggested commands", e)
            emptyList()
        }
        val historyItems = getHistoryCommand().filter { historyCommand ->
            suggestedItems.find {
                it.value == historyCommand.command.replace(
                    "\n",
                    ""
                )
            } == null
        }.map {
            CommandListItem(
                label = it.command.replace("\n", ""),
                value = it.command.replace("\n", ""),
                icon = AllIcons.Vcs.History,
                description = null,
                canDelete = true
            )
        }
        return suggestedItems + historyItems
    }
}
