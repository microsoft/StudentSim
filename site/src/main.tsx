import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

type PageDoc = {
  lang: string;
  title: string;
  description: string;
  style: string;
  markup: string;
};

const pages = {
  en: () => import('./pages/pagev4.html?raw'),
  zh: () => import('./pages/pagev4_zh.html?raw')
};

function extractTag(raw: string, tag: string): string {
  return raw.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, 'i'))?.[1].trim() ?? '';
}

function extractMetaDescription(raw: string): string {
  return raw.match(/<meta\s+name=["']description["']\s+content=["']([^"']*)["']/i)?.[1] ?? '';
}

function extractHtmlLang(raw: string, fallback: string): string {
  return raw.match(/<html[^>]*\slang=["']([^"']*)["']/i)?.[1] ?? fallback;
}

function parsePage(raw: string, fallbackLang: string): PageDoc {
  const body = extractTag(raw, 'body') || raw;

  return {
    lang: extractHtmlLang(raw, fallbackLang),
    title: extractTag(raw, 'title') || 'StudentSim',
    description: extractMetaDescription(raw),
    style: extractTag(raw, 'style'),
    markup: body.replace(/<script[\s\S]*?<\/script>/gi, '').trim()
  };
}

function setLanguageLinks(container: HTMLElement): void {
  const sectionHash = window.location.hash && !window.location.hash.startsWith('#/')
    ? window.location.hash
    : '';
  const routes: Record<string, string> = {
    './pagev4.html': `?lang=en${sectionHash}`,
    'pagev4.html': `?lang=en${sectionHash}`,
    './pagev4_zh.html': `?lang=zh${sectionHash}`,
    'pagev4_zh.html': `?lang=zh${sectionHash}`
  };

  container.querySelectorAll<HTMLAnchorElement>('a[href]').forEach((anchor) => {
    const href = anchor.getAttribute('href');
    if (href && routes[href]) {
      anchor.href = routes[href];
    }
  });
}

function bindCopyButtons(container: HTMLElement): () => void {
  const controllers: Array<() => void> = [];

  container.querySelectorAll<HTMLButtonElement>('button.copy').forEach((button) => {
    const handleClick = async () => {
      const code = button.parentElement?.querySelector('pre')?.textContent ?? '';
      try {
        await navigator.clipboard.writeText(code);
      } catch {
        const area = document.createElement('textarea');
        area.value = code;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        area.remove();
      }

      const original = button.textContent ?? 'Copy';
      button.textContent = 'Copied';
      window.setTimeout(() => {
        button.textContent = original;
      }, 1200);
    };

    button.addEventListener('click', handleClick);
    controllers.push(() => button.removeEventListener('click', handleClick));
  });

  return () => controllers.forEach((dispose) => dispose());
}

function scrollToCurrentSection(): void {
  if (!window.location.hash || window.location.hash.startsWith('#/')) {
    return;
  }

  document.getElementById(window.location.hash.slice(1))?.scrollIntoView();
}

function StaticPage({ raw, fallbackLang }: { raw: string; fallbackLang: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const page = useMemo(() => parsePage(raw, fallbackLang), [fallbackLang, raw]);

  useEffect(() => {
    document.documentElement.lang = page.lang;
    document.title = page.title;

    let description = document.querySelector<HTMLMetaElement>('meta[name="description"]');
    if (!description) {
      description = document.createElement('meta');
      description.name = 'description';
      document.head.appendChild(description);
    }
    description.content = page.description;
  }, [page.description, page.lang, page.title]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    setLanguageLinks(container);
    window.requestAnimationFrame(scrollToCurrentSection);
    window.setTimeout(scrollToCurrentSection, 250);
    window.setTimeout(scrollToCurrentSection, 1000);
    return bindCopyButtons(container);
  }, [page.markup]);

  return (
    <>
      <style>{page.style}</style>
      <div ref={containerRef} dangerouslySetInnerHTML={{ __html: page.markup }} />
    </>
  );
}

function getPageKey(): 'en' | 'zh' {
  const search = new URLSearchParams(window.location.search);
  const hash = window.location.hash.replace(/^#/, '');

  if (search.get('lang') === 'zh' || hash === '/zh' || hash.startsWith('/zh/')) {
    return 'zh';
  }

  return 'en';
}

function App() {
  const [pageKey, setPageKey] = useState(getPageKey);
  const [rawPage, setRawPage] = useState<string | null>(null);

  useEffect(() => {
    const syncPageKey = () => setPageKey(getPageKey());

    window.addEventListener('hashchange', syncPageKey);
    window.addEventListener('popstate', syncPageKey);
    return () => {
      window.removeEventListener('hashchange', syncPageKey);
      window.removeEventListener('popstate', syncPageKey);
    };
  }, []);

  const isZh = pageKey === 'zh';

  useEffect(() => {
    let active = true;

    pages[pageKey]().then((module) => {
      if (active) {
        setRawPage(module.default);
      }
    });

    return () => {
      active = false;
    };
  }, [pageKey]);

  if (!rawPage) {
    return null;
  }

  return <StaticPage raw={rawPage} fallbackLang={isZh ? 'zh-CN' : 'en'} />;
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
