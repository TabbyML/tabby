import { extensions } from "vscode";
import { API } from "./types/git";

const gitExt = extensions.getExtension("vscode.git");
export const gitApi: API = gitExt?.isActive ? gitExt.exports.getAPI(1) : undefined;
