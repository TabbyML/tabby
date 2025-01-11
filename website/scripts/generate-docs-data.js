const fs = require("fs");
const path = require("path");
const matter = require("gray-matter");

const docsDir = path.resolve(__dirname, "../docs");
const outputFile = path.resolve(__dirname, "../src/data/docsData.json");

function cleanDescription(content) {
  return content
    .replace(/!\[.*?\]\(.*?\)/g, "") // Remove images
    .replace(/\[([^\]]+)\]\(.*?\)/g, "$1") // Keep only the text from links
    .replace(/`([^`]+)`/g, "$1") // Remove inline code backticks
    .replace(/[#*>`~\-_\+]+/g, "") // Remove special characters (e.g., headings, blockquotes)
    .replace(/^\s*[\r\n]/gm, "") // Remove empty lines
    .trim();
}

function getAllDocs(dir) {
  const files = fs.readdirSync(dir);
  let docs = [];

  files.forEach((file) => {
    const fullPath = path.join(dir, file);

    if (fs.statSync(fullPath).isDirectory()) {
      docs = docs.concat(getAllDocs(fullPath)); // Recursively process subdirectories
    } else if (file.endsWith(".md") || file.endsWith(".mdx")) {
      const content = fs.readFileSync(fullPath, "utf-8");
      const { data, content: body } = matter(content); // Extract frontmatter and content

      const title =
        data.title || body.match(/^#\s+(.*)/m)?.[1] || "Untitled Document";

      const firstParagraph = body
        .split("\n")
        .find(
          (line) =>
            line.trim() && !line.startsWith("#") && !line.startsWith("---")
        );
      const description = cleanDescription(
        data.description || firstParagraph || "No description available."
      );

      docs.push({
        id: fullPath.replace(docsDir + "/", "").replace(/\.(md|mdx)$/, ""),
        path: `/docs/${fullPath
          .replace(docsDir + "/", "")
          .replace(/\.(md|mdx)$/, "")}`,
        title: title.trim(),
        description: description.trim(),
      });
    }
  });

  return docs;
}

const ensureDirectoryExistence = (filePath) => {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const docsData = getAllDocs(docsDir);
ensureDirectoryExistence(outputFile);
fs.writeFileSync(outputFile, JSON.stringify(docsData, null, 2));
console.log(`Docs data generated successfully at ${outputFile}`);
