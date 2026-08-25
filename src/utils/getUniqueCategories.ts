import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

interface Category {
  category: string;
  categoryName: string;
}

const getUniqueCategories = (posts: CollectionEntry<"blog">[]) => {
  const categories: Category[] = posts
    .filter(postFilter)
    .map(post => post.data.category)
    .filter((category): category is string => Boolean(category))
    .map(category => ({
      category: slugifyStr(category),
      categoryName: category,
    }))
    .filter(
      (value, index, self) =>
        self.findIndex(item => item.category === value.category) === index
    )
    .sort((a, b) => a.category.localeCompare(b.category));
  return categories;
};

export default getUniqueCategories;
