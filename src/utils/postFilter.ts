import type { CollectionEntry } from "astro:content";
import { SITE } from "@/config";

// 站点时区（Asia/Shanghai = UTC+8）的 UTC 偏移，用于解释仅含日期的 pubDatetime
const SITE_UTC_OFFSET_MS = 8 * 60 * 60 * 1000;

const postFilter = ({ data }: CollectionEntry<"blog">) => {
  const pubDate = new Date(data.pubDatetime);
  // 仅提供日期（如 2026-08-26）时，JS 会解析为 UTC 零点；
  // 按站点时区解释为当天零点，避免东八区当天凌晨被误判为未发布
  const isDateOnly =
    pubDate.getUTCHours() === 0 &&
    pubDate.getUTCMinutes() === 0 &&
    pubDate.getUTCSeconds() === 0 &&
    pubDate.getUTCMilliseconds() === 0;

  const pubTime = pubDate.getTime() + (isDateOnly ? SITE_UTC_OFFSET_MS : 0);
  const isPublishTimePassed = Date.now() > pubTime - SITE.scheduledPostMargin;
  return !data.draft && (import.meta.env.DEV || isPublishTimePassed);
};

export default postFilter;
