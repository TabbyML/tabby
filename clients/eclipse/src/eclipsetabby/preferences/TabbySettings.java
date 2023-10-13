package eclipsetabby.preferences;

import org.eclipse.jface.preference.StringFieldEditor;
import org.eclipse.jface.preference.PreferencePage;
import org.eclipse.jface.resource.ImageDescriptor;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.ui.IWorkbench;
import org.eclipse.ui.IWorkbenchPreferencePage;

public class TabbySettings extends PreferencePage implements IWorkbenchPreferencePage {

	public TabbySettings() {
		// TODO Auto-generated constructor stub
	}

	public TabbySettings(String title) {
		super(title);
		// TODO Auto-generated constructor stub
	}

	public TabbySettings(String title, ImageDescriptor image) {
		super(title, image);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void init(IWorkbench workbench) {
		// TODO Auto-generated method stub

	}

	@Override
	protected Control createContents(Composite parent) {
		Composite composite = new Composite(parent, SWT.NULL);
		StringFieldEditor stf = new StringFieldEditor("name","name",50, 1, composite);
		stf.setStringValue(PreferenceInitializer.getPreferenceStore().getString(PreferenceConstants.P_SERVER_ENDPOINT));
		return composite;
	}


}
