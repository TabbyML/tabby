package com.tabbyml.intellijtabby.inlineChat

import com.intellij.codeInsight.intention.impl.BaseIntentionAction;
import com.intellij.codeInsight.intention.preview.IntentionPreviewInfo
import com.intellij.icons.AllIcons
import com.intellij.ide.ui.LafManagerListener
import com.intellij.openapi.components.serviceOrNull
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.EditorCustomElementRenderer
import com.intellij.openapi.editor.Inlay
import com.intellij.openapi.editor.colors.EditorColorsManager
import com.intellij.openapi.editor.colors.EditorColorsScheme
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.fileEditor.FileDocumentManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.popup.PopupStep
import com.intellij.openapi.ui.popup.util.BaseListPopupStep
import com.intellij.psi.PsiFile
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.UIUtil
import java.awt.*
import java.awt.event.*
import javax.swing.*
import com.intellij.ui.components.IconLabelButton
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.Location
import org.eclipse.lsp4j.Position
import org.eclipse.lsp4j.Range

class InlineChatIntentionAction : BaseIntentionAction() {
    private var inlay: Inlay<InlineChatInlayRenderer>? = null
    private var inlayRender: InlineChatInlayRenderer? = null
    private var project: Project? = null
    private var location: Location? = null
    private var editor: Editor? = null
    override fun getFamilyName(): String {
        return "Tabby"
    }

    override fun isAvailable(project: Project, editor: Editor?, file: PsiFile?): Boolean {
        return true;
    }

    override fun invoke(project: Project, editor: Editor?, file: PsiFile?) {
        this.project = project
        this.editor = editor
        if (editor != null) {
            this.location = getCurrentLocation(editor = editor)
            addInputToEditor(editor = editor, offset = editor.caretModel.offset);
        }

        project.messageBus.connect().subscribe(LafManagerListener.TOPIC, LafManagerListener {
            inlayRender?.repaint() // FIXME
        })
    }

    private fun getCurrentLocation(editor: Editor): Location {
        val file = editor.document.let {
            FileDocumentManager.getInstance().getFile(it)
        }
        val fileUri = file?.let { "file://${it.path}" }
        val location = Location()
        location.uri = fileUri
        val selectionModel = editor.selectionModel
        val document = editor.document
        val caretOffset = editor.caretModel.offset
        var startOffset = caretOffset
        var endOffset = caretOffset
        if (selectionModel.hasSelection()) {
            startOffset = selectionModel.selectionStart
            endOffset = selectionModel.selectionEnd
        }
        val startPosition = Position(document.getLineNumber(startOffset), 0)
        val endChar = endOffset - document.getLineStartOffset(document.getLineNumber(endOffset))
        val endLine = if (endChar == 0) document.getLineNumber(endOffset) else document.getLineNumber(endOffset) + 1
        val endPosition = Position(endLine, endChar)
        location.range = Range(startPosition, endPosition)

        return location
    }


    override fun getText(): String {
        return "Open Tabby inline chat";
    }

    override fun generatePreview(project: Project, editor: Editor, file: PsiFile): IntentionPreviewInfo {
        // Return a custom preview or EMPTY to indicate no preview
        return IntentionPreviewInfo.EMPTY
    }

    private fun addInputToEditor(editor: Editor, offset: Int) {
        val inlayModel = editor.inlayModel
        inlayRender = InlineChatInlayRenderer(editor, this::onClose, this::onInputSubmit)
        inlay = inlayModel.addBlockElement(offset, true, true, 0, inlayRender!!)
    }

    private fun onClose() {
        inlay?.dispose()
    }

    private fun onInputSubmit(value: String) {
        chatEdit(command = value)
        editor?.selectionModel?.removeSelection()
    }

    private fun chatEdit(command: String) {
        val scope = CoroutineScope(Dispatchers.IO)
        scope.launch {
            val server = project?.serviceOrNull<ConnectionService>()?.getServerAsync() ?: return@launch
            if (location == null) {
                return@launch
            }
            val param = ChatEditParams(
                location = location!!,
                command = command
            )
            server.chatFeature.chatEdit(params = param)
        }
    }
}

class InlineChatInlayRenderer(
    private val editor: Editor,
    private val onClose: () -> Unit,
    private val onSubmit: (value: String) -> Unit
) :
    EditorCustomElementRenderer {
    private val inlineChatComponent = InlineChatComponent(onClose = this::removeComponent, onSubmit = onSubmit)
    private var targetRegion: Rectangle? = null

    init {
        setupEvent()
    }

    private fun setupEvent() {
        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(object : KeyEventDispatcher {
            override fun dispatchKeyEvent(e: KeyEvent): Boolean {
                if (e.id == KeyEvent.KEY_PRESSED && (e.keyCode == KeyEvent.VK_BACK_SPACE || e.keyCode == KeyEvent.VK_DELETE)) {
                    // Handle backspace and delete
                    if (e.component is JBTextArea) {
                        // Return true to consume the event (prevent default handling)
                        return true
                    }
                }
                if (e.id == KeyEvent.KEY_PRESSED && e.keyCode == KeyEvent.VK_ESCAPE) {
                    // Handle escape
                    removeComponent()
                    return true
                }
                return false
            }
        })
    }

    override fun calcWidthInPixels(inlay: Inlay<*>): Int {
        return inlineChatComponent.preferredSize.width
    }

    override fun calcHeightInPixels(inlay: Inlay<*>): Int {
        return inlineChatComponent.preferredSize.height
    }

    override fun paint(inlay: Inlay<*>, g: Graphics, targetRegion: Rectangle, textAttributes: TextAttributes) {
        if (this.targetRegion == null) {
            this.targetRegion = targetRegion
        }
        val firstTargetRegion = this.targetRegion ?: targetRegion
        inlineChatComponent.setSize(firstTargetRegion.width, firstTargetRegion.height)
        inlineChatComponent.setLocation(firstTargetRegion.x, firstTargetRegion.y)

        if (inlineChatComponent.parent == null) {
            editor.contentComponent.add(inlineChatComponent)
            inlineChatComponent.requestFocus()
        }

        inlineChatComponent.repaint()
    }

    fun repaint() {
        inlineChatComponent.repaint()
    }

    private fun removeComponent() {
        editor.contentComponent.remove(inlineChatComponent)
        editor.contentComponent.repaint();
        onClose()
    }
}

class InlineChatComponent(private val onClose: () -> Unit, private val onSubmit: (value: String) -> Unit) : JPanel() {
    private val closeButton = createCloseButton()
    private val inlineInput = InlineInputComponent(onSubmit = this::handleSubmit, onCancel = this::handleClose)

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
        closeButton.toolTipText = "Close Tabby inline chat"
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

class InlineInputComponent(private var onSubmit: (value: String) -> Unit, private var onCancel: () -> Unit) : JPanel() {
    private val textArea: JTextArea = createTextArea()
    private val submitButton: JLabel = createSubmitButton()
    private val historyButton: JLabel = createHistoryButton()

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
                    // Handle backspace and delete
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
//                println("textarea keyPressed")
            }

            override fun keyReleased(e: KeyEvent) {
                if (e.keyCode == KeyEvent.VK_BACK_SPACE || e.keyCode == KeyEvent.VK_DELETE) {
                    // Handle backspace and delete
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

    // Request focus for the text area
    override fun requestFocus() {
        textArea.requestFocus()
    }

    private fun handleConfirm() {
        onSubmit(textArea.text)
        textArea.text = ""
    }

    private fun createSubmitButton(): JLabel {
        val submitButton = IconLabelButton(AllIcons.Actions.Commit) { handleConfirm() }
        submitButton.toolTipText = "Submit the command"
        submitButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        submitButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)
        return submitButton
    }

    private fun handleOpenHistory() {
        val historyItems = listOf("History command 1", "History command 2", "/doc")
        val popup = JBPopupFactory.getInstance().createPopupChooserBuilder(historyItems)
            .setRenderer(object : DefaultListCellRenderer() {
                override fun getListCellRendererComponent(
                    list: JList<*>?,
                    value: Any?,
                    index: Int,
                    isSelected: Boolean,
                    cellHasFocus: Boolean
                ): Component {
                    val panel = JPanel(BorderLayout())
                    panel.preferredSize = Dimension(730, 20)

                    val label = JLabel(value.toString())
                    label.border = BorderFactory.createEmptyBorder(0, 10, 0, 10)

                    val deleteButton = IconLabelButton(AllIcons.Actions.Close) {
                        // Handle delete action here in a real implementation
                    }
                    deleteButton.border = BorderFactory.createEmptyBorder(0, 5, 0, 5)

                    panel.add(label, BorderLayout.CENTER)
                    panel.add(deleteButton, BorderLayout.EAST)

                    if (isSelected) {
                        panel.background = UIUtil.getListSelectionBackground(true)
                        label.foreground = UIUtil.getListSelectionForeground(true)
                    } else {
                        panel.background = UIUtil.getListBackground()
                        label.foreground = UIUtil.getListForeground()
                    }

                    return panel
                }
            })
            .setItemChosenCallback { selectedValue ->
                textArea.text = selectedValue
            }
            .createPopup()

        popup.showUnderneathOf(this)
    }

    private fun createHistoryButton(): JLabel {
        val historyButton = IconLabelButton(AllIcons.Actions.SearchWithHistory) { handleOpenHistory() }
        historyButton.toolTipText = "Select predefined / history Command"
        historyButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        historyButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)
        return historyButton
    }
}
