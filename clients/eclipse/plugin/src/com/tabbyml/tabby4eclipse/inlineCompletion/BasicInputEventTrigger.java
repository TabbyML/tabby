package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.HashMap;
import java.util.Map;

import org.eclipse.swt.custom.StyledText;
import org.eclipse.swt.events.KeyEvent;
import org.eclipse.swt.events.KeyListener;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.swt.events.MouseListener;
import org.eclipse.ui.texteditor.ITextEditor;

import com.tabbyml.tabby4eclipse.Logger;
import com.tabbyml.tabby4eclipse.editor.EditorUtils;

/**
 * This DocumentBasedTrigger listens to keyboard and mouse events to determine
 * when to trigger inline completion. When a keyboard or mouse key up event is
 * received, check if the cursor is moved, then check if document has changes,
 * and trigger inline completion if so, otherwise dismiss inline completion.
 */
public class BasicInputEventTrigger implements IInlineCompletionTrigger {
	private Logger logger = new Logger("InlineCompletionTrigger.BasicInputEventTrigger");

	private IInlineCompletionService inlineCompletionService = InlineCompletionService.getInstance();

	private Map<ITextEditor, KeyListener> keyListeners = new HashMap<>();
	private Map<ITextEditor, MouseListener> mouseListeners = new HashMap<>();

	private int lastOffset = -1;
	private long lastModificationStamp = -1;

	@Override
	public void register(ITextEditor textEditor) {
		StyledText widget = EditorUtils.getStyledTextWidget(textEditor);
		widget.getDisplay().syncExec(() -> {
			KeyListener keyListener = new KeyListener() {
				@Override
				public void keyPressed(KeyEvent e) {
				}

				@Override
				public void keyReleased(KeyEvent e) {
					handleEvent();
				}
			};
			widget.addKeyListener(keyListener);
			keyListeners.put(textEditor, keyListener);
			logger.debug("Created key listener for TextEditor " + textEditor.toString());

			MouseListener mouseListener = new MouseListener() {
				@Override
				public void mouseDoubleClick(MouseEvent e) {
				}

				@Override
				public void mouseDown(MouseEvent e) {
				}

				@Override
				public void mouseUp(MouseEvent e) {
					handleEvent();
				}
			};
			widget.addMouseListener(mouseListener);
			mouseListeners.put(textEditor, mouseListener);
			logger.debug("Created mouse listener for TextEditor " + textEditor.toString());
		});
	}

	@Override
	public void unregister(ITextEditor textEditor) {
		StyledText widget = EditorUtils.getStyledTextWidget(textEditor);
		widget.getDisplay().syncExec(() -> {
			KeyListener keyListener = keyListeners.get(textEditor);
			if (keyListener != null) {
				widget.removeKeyListener(keyListener);
				keyListeners.remove(textEditor);
				logger.debug("Removed key listener for TextEditor " + textEditor.toString());
			}
			MouseListener mouseListener = mouseListeners.get(textEditor);
			if (mouseListener != null) {
				widget.removeMouseListener(mouseListener);
				mouseListeners.remove(textEditor);
				logger.debug("Removed mouse listener for TextEditor " + textEditor.toString());
			}
		});
	}

	private void handleEvent() {
		logger.debug("handle input event.");
		ITextEditor textEditor = EditorUtils.getActiveTextEditor();
		if (textEditor == null) {
			return;
		}
		int offset = EditorUtils.getCurrentOffsetInDocument(textEditor);
		long modificationStamp = EditorUtils.getDocumentModificationStamp(textEditor);
		if (lastOffset != -1 && lastModificationStamp != -1) {
			if (offset != lastOffset) {
				logger.debug("offset cahnged, check next...");
				if (modificationStamp != lastModificationStamp) {
					logger.debug("modificationStamp changed, trigger inline completion.");
					inlineCompletionService.trigger(false);
				} else {
					logger.debug("modificationStamp not changed, dismiss inline completion.");
					inlineCompletionService.dismiss();
				}
			} else {
				logger.debug("offset not changed, ignore event.");
			}
		} else {
			logger.debug("init offset and modificationStamp.");
		}
		lastOffset = offset;
		lastModificationStamp = modificationStamp;
	}
}
