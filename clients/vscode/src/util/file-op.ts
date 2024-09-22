import fs from "fs";
import path from "path";

export function listDir(p: string) {
  const dirs = fs.readdirSync(p);
  const result = [];

  for (const dir of dirs) {
    const resolved = path.resolve(p, dir);
    result.push({
      description: resolved,
      label: dir,
    });
  }

  return result;
}

export function readStatSync(filename: string) {
  return fs.statSync(filename);
}
