package com.tabbyml.codecompletion.settings;

import org.eclipse.core.runtime.preferences.AbstractPreferenceInitializer;
import org.eclipse.core.runtime.preferences.InstanceScope;
import org.eclipse.jface.preference.IPreferenceStore;
import org.eclipse.ui.preferences.ScopedPreferenceStore;
import org.osgi.framework.FrameworkUtil;


/**
 * Class used to initialize default preference values.
 */
public class PreferenceInitializer extends AbstractPreferenceInitializer {
	
	public static String PLUGIN_ID = "tabby-eclipse";
	private static final IPreferenceStore PREFERENCE_STORE = new ScopedPreferenceStore(InstanceScope.INSTANCE,
			FrameworkUtil.getBundle(PreferenceInitializer.class).getSymbolicName());

	public static IPreferenceStore getPreferenceStore() {
		return PREFERENCE_STORE;
	}
	/*
	 * (non-Javadoc)
	 * 
	 * @see org.eclipse.core.runtime.preferences.AbstractPreferenceInitializer#initializeDefaultPreferences()
	 */
	@Override
	public void initializeDefaultPreferences() {
		
		getPreferenceStore().setDefault(PreferenceConstants.P_SERVER_ENDPOINT, "http://localhost:8080");

	}
	

}
