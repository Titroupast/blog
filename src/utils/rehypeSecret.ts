import type { Element, ElementContent, Root, RootContent } from "hast";

const SECRET_MARKERS = ["🔒", "🤫", "secret", "保密", "秘密", "内部", "私有"];
const HEADING_RE = /^h[1-6]$/;

function elementText(node: Element): string {
  let text = "";
  for (const child of node.children) {
    if (child.type === "text") {
      text += child.value;
    } else if (child.type === "element") {
      text += elementText(child);
    }
  }
  return text;
}

function isSecretHeading(node: Element): boolean {
  if (!HEADING_RE.test(node.tagName)) return false;
  const text = elementText(node).toLowerCase();
  return SECRET_MARKERS.some(marker => text.includes(marker.toLowerCase()));
}

function headingDepth(node: Element): number {
  return Number(node.tagName.slice(1));
}

function createSecretDiv(children: (ElementContent | RootContent)[]): Element {
  return {
    type: "element",
    tagName: "div",
    properties: {
      className: ["secret-only"],
      "data-pagefind-ignore": "",
    },
    children: children as ElementContent[],
  };
}

function findSecretGroup(
  children: (ElementContent | RootContent)[],
  start: number
): { group: (ElementContent | RootContent)[]; nextIndex: number } | null {
  const first = children[start];
  if (first?.type !== "element" || !isSecretHeading(first)) return null;

  const depth = headingDepth(first);
  const group: (ElementContent | RootContent)[] = [first];
  let nextIndex = start + 1;

  while (nextIndex < children.length) {
    const next = children[nextIndex];
    if (
      next.type === "element" &&
      HEADING_RE.test(next.tagName) &&
      headingDepth(next) <= depth
    ) {
      break;
    }
    group.push(next);
    nextIndex += 1;
  }

  return { group, nextIndex };
}

function processChildren<T extends ElementContent | RootContent>(
  children: T[]
): T[] {
  const result: T[] = [];
  let i = 0;

  while (i < children.length) {
    const child = children[i];

    if (child.type === "element") {
      child.children = processChildren(child.children);
    }

    const secretGroup = findSecretGroup(children, i);
    if (secretGroup) {
      result.push(createSecretDiv(secretGroup.group) as unknown as T);
      i = secretGroup.nextIndex;
      continue;
    }

    result.push(child);
    i += 1;
  }

  return result;
}

export default function rehypeSecret() {
  return (tree: Root) => {
    tree.children = processChildren(tree.children);
  };
}
