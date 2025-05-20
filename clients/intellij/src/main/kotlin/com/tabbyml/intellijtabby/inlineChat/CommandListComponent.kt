package com.tabbyml.intellijtabby.inlineChat

import com.intellij.icons.AllIcons
import com.intellij.openapi.application.ApplicationManager
import com.intellij.ui.CollectionListModel
import com.intellij.ui.components.IconLabelButton
import com.intellij.ui.components.JBLabel
import com.intellij.ui.components.JBList
import com.intellij.ui.components.JBScrollPane
import com.intellij.util.ui.JBUI
import com.intellij.util.ui.UIUtil
import java.awt.*
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.util.function.Consumer
import javax.swing.*


data class CommandListItem(
    var label: String,
    var value: String,
    var icon: Icon,
    var description: String?,
    val canDelete: Boolean
)

class CommandListComponent(
    private val title: String = "Commands",
    initialData: List<CommandListItem>?,
    private val onItemSelected: Consumer<CommandListItem>?,
    private val onItemDeleted: Consumer<CommandListItem>?,
    private val onClearAll: () -> Unit,
) {
    val list: JBList<CommandListItem>
    private val model: CollectionListModel<CommandListItem> = CollectionListModel(initialData ?: emptyList())
    private val scrollPane: JBScrollPane
    private val mainPanel: JPanel = JPanel(BorderLayout())

    private var hoveredIndex: Int = -1

    init {
        list = JBList(model)
        list.cellRenderer = CustomListItemRenderer { hoveredIndex }
        list.selectionMode = ListSelectionModel.SINGLE_SELECTION
        list.setEmptyText("No items")

        list.addMouseListener(object : MouseAdapter() {
            override fun mouseClicked(e: MouseEvent) {
                handleMouseAction(e)
            }

            override fun mouseExited(e: MouseEvent) {
                if (hoveredIndex != -1) {
                    hoveredIndex = -1
                    list.repaint()
                }
            }
        })

        list.addMouseMotionListener(object : MouseAdapter() {
            override fun mouseMoved(e: MouseEvent) {
                val index = list.locationToIndex(e.point)
                if (index != hoveredIndex) {
                    hoveredIndex = index
                    list.repaint()
                }
            }
        })
        val toolbar = createToolbar()
        scrollPane = JBScrollPane(list)
        mainPanel.add(toolbar, BorderLayout.NORTH)
        mainPanel.add(scrollPane, BorderLayout.CENTER)
    }

    private fun createToolbar(): JPanel {
        val toolbar = JPanel(BorderLayout()).apply {
            preferredSize = Dimension(730, 20)
        }
        toolbar.border = JBUI.Borders.empty(3, 5)
        val titleLabel = JBLabel(title)
        titleLabel.font = JBUI.Fonts.label().deriveFont(JBUI.Fonts.label().size + 1.0f)
//        val clearAllButton = IconLabelButton(AllIcons.Actions.GC) {
//            onClearAll()
//        }
//        clearAllButton.toolTipText = "Clear all commands"
//        clearAllButton.cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
//        clearAllButton.border = BorderFactory.createEmptyBorder(4, 8, 4, 8)

        toolbar.add(titleLabel, BorderLayout.WEST)
//        toolbar.add(clearAllButton, BorderLayout.EAST)

        return toolbar
    }

    private fun handleMouseAction(e: MouseEvent) {
        val index = list.locationToIndex(e.point)
        if (index != -1) {
            val item = model.getElementAt(index)
            val deleteBounds = CustomListItemRenderer.getDeleteButtonBounds(list, index, item)

            if (deleteBounds != null && deleteBounds.contains(e.point)) {
                e.consume() // Prevent list selection change on delete click
                onItemDeleted?.accept(item)
            } else {
                val selectedValue = list.selectedValue
                if (selectedValue != null) {
                    onItemSelected?.accept(selectedValue)
                }
            }
        }
    }

    fun setData(newData: List<CommandListItem>, onUpdated: (() -> Unit)? = null) {
        ApplicationManager.getApplication().invokeLater {
            val selectedValue = list.selectedValue
            model.replaceAll(newData)
            if (selectedValue != null) {
                val newIndex = model.getElementIndex(selectedValue)
                if (newIndex != -1) {
                    list.setSelectedIndex(newIndex)
                } else {
                    list.clearSelection()
                }
            } else {
                list.clearSelection()
            }
            onUpdated?.invoke()
        }
    }

    val component: JComponent
        get() = mainPanel

    fun dispose() {
        mainPanel.removeAll()
        scrollPane.viewport.view = null
        list.model = CollectionListModel()
    }
}


class CustomListItemRenderer(private val getHoveredIndex: () -> Int) : JPanel(), ListCellRenderer<CommandListItem> {
    private val iconLabel: JBLabel
    private val contentPanel: JPanel
    private val label: JLabel
    private val desc: JLabel
    private val deleteButton: JLabel

    init {
        layout = BorderLayout(JBUI.scale(5), 0)
        border = JBUI.Borders.empty(2, 5)

        iconLabel = JBLabel()
        label = JLabel()
        desc = JLabel()
        contentPanel = JPanel(BorderLayout())
        contentPanel.add(desc, BorderLayout.CENTER)
        contentPanel.add(label, BorderLayout.WEST)
        deleteButton = JLabel(DELETE_ICON).apply {
            isOpaque = false
            toolTipText = "delete command"
            preferredSize =
                Dimension(DELETE_ICON.iconWidth, DELETE_ICON.iconHeight)
        }

        add(deleteButton, BorderLayout.EAST)
        add(iconLabel, BorderLayout.WEST)
        add(contentPanel, BorderLayout.CENTER)
    }

    override fun getListCellRendererComponent(
        list: JList<out CommandListItem>,
        value: CommandListItem,
        index: Int,
        isSelected: Boolean,
        cellHasFocus: Boolean
    ): Component {
        iconLabel.icon = value.icon
        label.text = value.label
        desc.text = value.description
        label.border = BorderFactory.createEmptyBorder(0, 10, 0, 10)
        desc.foreground = UIUtil.getContextHelpForeground()
        contentPanel.isOpaque = false

        val isHovered = index == getHoveredIndex()

        if (isSelected) {
            background = UIUtil.getListSelectionBackground(true) // Use focus-aware color
            iconLabel.foreground = UIUtil.getListSelectionForeground(true)
        } else if (isHovered) {
            background = UIUtil.getListSelectionBackground(false)
            iconLabel.foreground = UIUtil.getListForeground()
            cursor = Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)
        } else {
            background = UIUtil.getListBackground()
            iconLabel.foreground = UIUtil.getListForeground()
            cursor = Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR)
        }

        isOpaque = true

        if (value.canDelete) {
            deleteButton.isVisible = true
            if (isHovered) {
                deleteButton.setIcon(DELETE_ICON_HOVERED);
            } else {
                deleteButton.setIcon(DELETE_ICON);
            }
        } else {
            deleteButton.isVisible = false
        }

        return this
    }

    companion object {
        val DELETE_ICON: Icon = AllIcons.Actions.Close
        val DELETE_ICON_HOVERED: Icon = AllIcons.Actions.CloseHovered

        fun getDeleteButtonBounds(list: JList<out CommandListItem>, index: Int, item: CommandListItem): Rectangle? {
            if (index < 0 || index >= list.model.size) {
                return null
            }
            val cellBounds = list.getCellBounds(index, index) ?: return null
            val renderer = CustomListItemRenderer { index }
            renderer.getListCellRendererComponent(list, item, index, true, false)
            val prefSize = renderer.preferredSize
            renderer.setBounds(0, 0, cellBounds.width, prefSize.height)
            renderer.doLayout()
            val deleteBoundsRelativeToPanel = renderer.deleteButton.bounds
            return Rectangle(
                cellBounds.x + deleteBoundsRelativeToPanel.x,
                cellBounds.y + deleteBoundsRelativeToPanel.y,
                deleteBoundsRelativeToPanel.width,
                deleteBoundsRelativeToPanel.height
            )
        }
    }
}

