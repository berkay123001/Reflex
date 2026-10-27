(() => {
  if (!document.body) return null;
  try {
    const adSel = 'ins.adsbygoogle, [id^="google_ads_iframe"], [id^="aswift_"], iframe[src*="googleads"], iframe[src*="doubleclick"], [aria-label="Advertisement"], [id*="google_vignette"]';
    document.querySelectorAll(adSel).forEach(el => {
      const wrapper = el.closest('ins.adsbygoogle, [id^="aswift_"]') || el;
      wrapper.remove();
    });
    if (document.body.style.overflow === 'hidden' && location.hash === '#google_vignette') {
      document.body.style.overflow = '';
      document.documentElement.style.overflow = '';
    }
  } catch (_) {}
  const cache = window.__jevFast ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  const safe = e => !['password','file','hidden'].includes(e.type);
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const name = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(document.getElementById(id),seen)).filter(Boolean).join(' ');
    const raw = referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && !['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','SVG'].includes(n.tagName) && n.getAttribute('aria-hidden')!=='true' ? name(n,seen) : '').join(' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
    return raw.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, '').replace(/\s+/g, ' ').trim();
  };
  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const challengeIframeSel='iframe[src*="challenges.cloudflare.com"],iframe[title*="Cloudflare"],iframe[title*="challenge" i],iframe[src*="recaptcha"],iframe[src*="hcaptcha"]';
  const selector='a[href],button,input,textarea,select,summary,[contenteditable="true"],'+
    challengeIframeSel+','+
    roles.map(role=>'[role="'+role+'"]').join(',');
  const role = e => {
    if (e.tagName==='IFRAME') return 'checkbox';
    const explicit=e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A') return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    [...document.querySelectorAll('input,textarea,select')].filter(safe)
      .map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),e.value??null,e.checked??null,e.selectedIndex??null,
      e.readOnly??null,e.matches(':disabled'),e.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      e.getAttribute('href'),scope?.innerText?.slice(0,6000)||''];
  };
  const clipRectCache=new Map();
  const getClipRect=el=>{
    if (!el || el===document.body || el===document.documentElement)
      return {top:0,left:0,bottom:innerHeight,right:innerWidth};
    if (clipRectCache.has(el)) return clipRectCache.get(el);
    const p=getClipRect(el.parentElement);
    const st=getComputedStyle(el);
    let top=p.top, left=p.left, bottom=p.bottom, right=p.right;
    if (['hidden','clip','auto','scroll'].includes(st.overflowY) || ['hidden','clip','auto','scroll'].includes(st.overflow)) {
      const cr=el.getBoundingClientRect();
      top=Math.max(top,cr.top); bottom=Math.min(bottom,cr.bottom);
    }
    if (['hidden','clip','auto','scroll'].includes(st.overflowX) || ['hidden','clip','auto','scroll'].includes(st.overflow)) {
      const cr=el.getBoundingClientRect();
      left=Math.max(left,cr.left); right=Math.min(right,cr.right);
    }
    const res={top,left,bottom,right};
    clipRectCache.set(el,res);
    return res;
  };
  let slotLikeCount=0;
  for (const b of document.querySelectorAll('button,[role="button"]')) {
    const sp=[...b.querySelectorAll('span')].map(s=>s.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
    const bt=(b.innerText||'').replace(/\s+/g,' ').trim();
    if (/^(?:[1-9]|10)$/.test(bt) || (sp.length===2 && /^(?:[1-9]|10)$/.test(sp[1]))) slotLikeCount++;
  }
  const allCards=[...document.querySelectorAll('article,section,li,[class*="Card-root"],[class*="card"]')].filter(
    c=>c.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]')
  );
  const isSerp=/(^|\.)(google|bing|duckduckgo)\./i.test(location.hostname) &&
    (location.pathname.includes('/search') || location.search.includes('q='));
  const seenCardLinks=new Set();
  const actions=[];
  for (const e of document.querySelectorAll(selector)) {
    if (!safe(e) || !visible(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
    const parentClickable=e.parentElement?.closest('a[href],button,[role="button"],[role="link"]');
    if (parentClickable && (name(parentClickable)||'').trim() && (name(parentClickable)||'').trim()===(name(e)||'').trim()) continue;
    const r=e.getBoundingClientRect(), rname=role(e);
    const clip=getClipRect(e.parentElement);
    const visLeft=Math.max(clip.left,r.left), visRight=Math.min(clip.right,r.right);
    const visTop=Math.max(clip.top,r.top), visBottom=Math.min(clip.bottom,r.bottom);
    const visW=visRight-visLeft, visH=visBottom-visTop;
    if (!rname || r.width<=0 || r.height<=0 || visW<Math.min(16,r.width*0.5) || visH<Math.min(20,r.height*0.35)) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    let lbl=(e.tagName==='IFRAME'?'Verify you are human (Security Challenge)':name(e))||rname;
    if (isSerp && /^(google ana sayfasına git|go to google home|harf çevirme aracı|sesle arama|görselle arama|hızlı ayarlar|google uygulamaları|geri bildirim)$/i.test(lbl.trim())) continue;
    if (rname==='button' && slotLikeCount>=3) {
      const spans=[...e.querySelectorAll('span')].map(s=>s.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
      if (spans.length===2 && /^(?:[1-9]|10)$/.test(spans[1])) {
        lbl=`[DOLU: ${spans[0]}] Slot ${spans[1]}`;
      } else {
        const m=/^(.+?)\s+([1-9]|10)$/.exec(lbl.replace(/\s+/g,' ').trim());
        if (m && !/^(?:slot|başla|tekrar)/i.test(m[1])) {
          lbl=`[DOLU: ${m[1]}] Slot ${m[2]}`;
        }
      }
    }
    if (rname==='link' && lbl.length>35) {
      const hEl=e.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]');
      const hTxt=hEl?.innerText?.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g,'').replace(/\s+/g,' ').trim();
      if (hTxt && hTxt.length>=4 && !lbl.toLowerCase().startsWith(hTxt.toLowerCase())) {
        lbl=`${hTxt} · ${lbl}`;
      }
    }
    if (/^(devamını oku|read more|detay|detaylar|incele|open|view|daha fazla)$/i.test(lbl.trim())) {
      let card=e.parentElement;
      while (card && card!==document.body) {
        if (card.matches('article,section,li,[class*="Card-root"],[class*="card"]') &&
            card.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]')) break;
        card=card.parentElement;
      }
      const hEl=card?.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]');
      const hTxt=hEl?.innerText?.replace(/\s+/g,' ').trim();
      const cIdx=card ? allCards.indexOf(card) : -1;
      const prefix=cIdx>=0 ? `#${cIdx+1}: ` : '';
      if (hTxt) lbl=`${lbl.trim()} (${prefix}${hTxt.slice(0,90)})`;
    }
    if (['radio','checkbox','textbox'].includes(rname)) {
      const qItem=e.closest('[role="listitem"],[role="group"],fieldset');
      const qHead=qItem?.querySelector('[role="heading"],legend,h2,h3,h4');
      const qTxt=qHead?.innerText?.replace(/\s*\*\s*$/,'').replace(/\s+/g,' ').trim();
      if (qTxt && !lbl.toLowerCase().includes(qTxt.slice(0,25).toLowerCase())) {
        lbl=`${qTxt.slice(0,75)} → ${lbl.trim()}`;
      }
    }
    const linkEl=e.tagName==='A'?e:e.closest('a[href]');
    const rawHref=(linkEl?.href && !linkEl.href.startsWith('javascript:')) ? linkEl.href.slice(0,240) : '';
    if (rname==='link' && rawHref) {
      const cardScope=e.closest('article,li,tr,[class*="Card-root"],[class*="card"]') || e.parentElement;
      const scopeId=cardScope ? identity(cardScope) : 0;
      const dedupKey=`${scopeId}|${rawHref}|${lbl.trim().toLowerCase()}`;
      if (seenCardLinks.has(dedupKey)) continue;
      seenCardLinks.add(dedupKey);
    }
    const base={node:identity(e),role:rname,label:lbl,
      rect:{x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)}};
    if (rawHref) base.href=rawHref;
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=value;
    }
    if (['checkbox','radio'].includes(e.type)) base.checked=String(e.checked);
    if (e.tagName==='SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        actions.push({...base,kind:'select',value:o.value,
          current_value:[...e.selectedOptions].map(o=>o.label).join(', '),label:base.label+' → '+o.label});
    } else {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      if (editable && rname==='combobox' && !value) actions.push({...base,kind:'click',value,label:'Open '+base.label});
    }
  }
  const clipCache=new Map();
  const getClipBottom=el=>{
    if (!el || el===document.body || el===document.documentElement) return Infinity;
    if (clipCache.has(el)) return clipCache.get(el);
    let b=getClipBottom(el.parentElement);
    const st=getComputedStyle(el);
    if (st.overflowY==='hidden' || st.overflowY==='clip' || st.overflow==='hidden' || st.overflow==='clip') {
      b=Math.min(b,el.getBoundingClientRect().bottom);
    }
    clipCache.set(el,b);
    return b;
  };
  const words=[], fullWords=[], walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  const range=document.createRange(); let node,length=0,fullLength=0;
  while ((node=walker.nextNode()) && (length<6000 || fullLength<30000)) {
    const value=node.textContent.trim(), parent=node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
    range.selectNodeContents(node); const r=range.getBoundingClientRect();
    if (r.width<=0 || r.height<=0 || r.top>=getClipBottom(parent)-2) continue;
    if (fullLength<30000) {
      fullWords.push(value); fullLength+=value.length;
    }
    if (length<6000 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
      words.push(value); length+=value.length;
    }
  }
  const isUnclipped=el=>{
    if (!visible(el)) return false;
    const r=el.getBoundingClientRect();
    return r.width>0 && r.height>0 && r.top<getClipBottom(el.parentElement)-2;
  };
  const h12=[...document.querySelectorAll('h1,h2')].filter(isUnclipped)
    .map(h=>`[${h.tagName.toLowerCase()}] `+h.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
  const h34=[...document.querySelectorAll('h3,h4')].filter(isUnclipped)
    .map(h=>`[${h.tagName.toLowerCase()}] `+h.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
  const headings=[...new Set([...h12,...h34])].slice(0,80);
  const text=words.join('\n').slice(0,6000), full_text=fullWords.join('\n').slice(0,30000), height=document.documentElement.scrollHeight;
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6]];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:420});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-420});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,full_text,headings,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
})()
