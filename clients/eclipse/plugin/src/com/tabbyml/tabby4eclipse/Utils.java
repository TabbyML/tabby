package com.tabbyml.tabby4eclipse;

import org.eclipse.swt.dnd.Clipboard;
import org.eclipse.swt.dnd.TextTransfer;
import org.eclipse.swt.dnd.Transfer;
import org.eclipse.swt.widgets.Display;

public class Utils {
	public static boolean isWindows() {
		return System.getProperty("os.name").toLowerCase().contains("win");
	}
	
    public static void setClipboardContent(String content) {
        Display display = Display.getCurrent();
        if (display == null) {
            display = Display.getDefault();
        }

        Clipboard clipboard = new Clipboard(display);
        TextTransfer textTransfer = TextTransfer.getInstance();
        clipboard.setContents(new Object[] { content }, new Transfer[] { textTransfer });
        clipboard.dispose();
    }
}
