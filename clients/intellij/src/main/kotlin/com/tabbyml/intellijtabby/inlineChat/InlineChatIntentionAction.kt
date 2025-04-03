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
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.psi.PsiFile
import com.intellij.ui.components.JBTextArea
import com.intellij.util.ui.UIUtil
import java.awt.*
import java.awt.event.*
import javax.swing.*
import com.intellij.ui.components.IconLabelButton
import com.intellij.openapi.ui.popup.JBPopupFactory
import com.intellij.openapi.ui.popup.JBPopupListener
import com.tabbyml.intellijtabby.lsp.ConnectionService
import com.tabbyml.intellijtabby.lsp.protocol.ChatEditParams
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.eclipse.lsp4j.Location
import org.eclipse.lsp4j.Range

class InlineChatIntentionAction : BaseIntentionAction(), DumbAware {
    private var inlay: Inlay<InlineChatInlayRenderer>? = null
    private var inlayRender: InlineChatInlayRenderer? = null
    private var project: Project? = null
    private var editor: Editor? = null
    override fun getFamilyName(): String {
        return "Tabby"
    }

    override fun isAvailable(project: Project, editor: Editor?, file: PsiFile?): Boolean {
        return true;
    }

    override fun invoke(project: Project, editor: Editor?, file: PsiFile?) {
        val inlineChatService = project.serviceOrNull<InlineChatService>() ?: return
        if (inlineChatService.inlineChatEditing) return
        inlineChatService.inlineChatEditing = true
        this.project = project
        this.editor = editor
        if (editor != null) {
            inlineChatService.location = getCurrentLocation(editor = editor)
            addInputToEditor(project, editor, editor.caretModel.offset);
        }

        project.messageBus.connect().subscribe(LafManagerListener.TOPIC, LafManagerListener {
            inlayRender?.repaint() // FIXME
        })
    }

    override fun getText(): String {
        return "Open Tabby inline chat";
    }

    override fun generatePreview(project: Project, editor: Editor, file: PsiFile): IntentionPreviewInfo {
        // Return a custom preview or EMPTY to indicate no preview
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
        inlineChatService.inlineChatEditing = false
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
    private val inlineChatComponent = InlineChatComponent(project, this::removeComponent, onSubmit)
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

class InlineChatComponent(private val project: Project, private val onClose: () -> Unit, private val onSubmit: (value: String) -> Unit) : JPanel() {
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

data class PickItem(var label: String, var value: String, var icon: Icon, var description: String?, val canDelete: Boolean)
data class ChatEditFileContext(val referrer: String, val uri: String, val range: Range)
data class InlineEditCommand(val command: String, val context: List<ChatEditFileContext>?)

class InlineInputComponent(private var project: Project, private var onSubmit: (value: String) -> Unit, private var onCancel: () -> Unit) : JPanel() {
    private val history: CommandHistory? = project.serviceOrNull<CommandHistory>()
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

    private fun createHistoryButton(): JLabel {
        val historyButton = IconLabelButton(AllIcons.Actions.SearchWithHistory) { handleOpenHistory() }
        historyButton.toolTipText = "Select suggested / history Command"
        historyButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        historyButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)
        return historyButton
    }

    private fun handleOpenHistory() {
        val commandItems = getCommandList()
        val popup = JBPopupFactory.getInstance().createPopupChooserBuilder<PickItem>(commandItems)
            .setRenderer(object : DefaultListCellRenderer() {
                override fun getListCellRendererComponent(
                    list: JList<*>?,
                    value: Any?,
                    index: Int,
                    isSelected: Boolean,
                    cellHasFocus: Boolean
                ): Component {
                    if (value !is PickItem) {
                        return super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus)
                    }
                    val panel = JPanel(BorderLayout())
                    panel.preferredSize = Dimension(730, 20)
                    val label = JLabel(value.label)
                    val desc = JLabel(value.description)
                    label.border = BorderFactory.createEmptyBorder(0, 10, 0, 10)
                    panel.add(JLabel(value.icon), BorderLayout.WEST)
                    val contentPanel = JPanel(BorderLayout())
                    contentPanel.add(label, BorderLayout.WEST)
                    desc.foreground = UIUtil.getContextHelpForeground()
                    contentPanel.add(desc, BorderLayout.CENTER)
                    contentPanel.isOpaque = false
                    panel.add(contentPanel, BorderLayout.CENTER)
                    if (value.canDelete) {
                        val deleteButton = IconLabelButton(AllIcons.Actions.Close) {
                            history?.deleteCommand(value.value)
                            handleOpenHistory()
                        }
                        deleteButton.toolTipText = "Delete this command from history"
//                        deleteButton.cursor =  Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
                        deleteButton.border = BorderFactory.createEmptyBorder(0, 5, 0, 5)
                        panel.add(deleteButton, BorderLayout.EAST)
                    }

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
                textArea.text = selectedValue.value
            }
            .createPopup()

        popup.showUnderneathOf(this)
    }

    private fun getHistoryCommand(): List<InlineEditCommand> {
        return history?.getHistory()?.map {
            InlineEditCommand(it, null)
        } ?: emptyList()
    }

    private fun getCommandList(): List<PickItem> {
        val location = project.serviceOrNull<InlineChatService>()?.location ?: return emptyList()
        val suggestedItems = getSuggestedCommands(project, location).get()?.map { PickItem(label = it.label, value = it.command, icon = AllIcons.Debugger.ThreadRunning, description = it.command, canDelete = false) } ?: emptyList()
        val historyItems = getHistoryCommand().filter {historyCommand -> suggestedItems.find { it.value == historyCommand.command.replace("\n", "") } == null }.map {
            PickItem(label = it.command.replace("\n", ""), value = it.command.replace("\n", ""), icon = AllIcons.Vcs.History, description = null, canDelete = true)
        }
        return suggestedItems + historyItems
    }
}
