import type { CollectionEntry } from "astro:content";
import getSortedPosts from "./getSortedPosts";
import { slugifyStr } from "./slugify";

const getPostsByCategory = (
  posts: CollectionEntry<"blog">[],
  category: string
) =>
  getSortedPosts(
    posts.filter(
      post =>
        Boolean(post.data.category) &&
        slugifyStr(post.data.category) === category
    )
  );

export default getPostsByCategory;
