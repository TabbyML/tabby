package com.tabbyml.tabby4eclipse.statusbar;

import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Menu;
import org.eclipse.swt.widgets.MenuItem;
import org.eclipse.swt.custom.CLabel;
import org.eclipse.swt.events.MouseAdapter;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.ui.ISharedImages;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.menus.WorkbenchWindowControlContribution;

import com.tabbyml.tabby4eclipse.Images;

public class StatusbarContribution extends WorkbenchWindowControlContribution {
	private CLabel label;

	@Override
	protected Control createControl(Composite parent) {
		label = new CLabel(parent, 0);
		label.setImage(Images.getIconCheck());
		label.setText("Tabby");
		label.setToolTipText("Tabby");
		label.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseUp(MouseEvent e) {
				Menu menu = new Menu(label);
				MenuItem item = new MenuItem(menu, 0);
				item.setText("Tabby plugin is active.");
				menu.setVisible(true);
			}
		});
		return label;
	}
}
