(() => {
  if (!document.body) return null;
  const allRoots = [document];
  const collectShadowRoots = start => {
    if (!start) return;
    const walker = document.createTreeWalker(start, NodeFilter.SHOW_ELEMENT);
    let el;
    while ((el = walker.nextNode())) {
      if (el.shadowRoot) {
        allRoots.push(el.shadowRoot);
        collectShadowRoots(el.shadowRoot);
      }
    }
  };
  collectShadowRoots(document.body);
  const cache = window.__jevFast ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  const safe = e => !['password','file','hidden'].includes(e.type);
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const choiceFallbackLabel = (e,seen) => {
    if (e.tagName!=='INPUT' || !['radio','checkbox'].includes(e.type)) return '';
    const cl=e.closest('label');
    if (cl) { const t=name(cl,seen); if (t) return t; }
    const p=e.parentElement;
    if (!p || ['BODY','HTML','FORM','FIELDSET'].includes(p.tagName)) return '';
    const sib=[...p.children].filter(c=>c.tagName==='LABEL').map(l=>name(l,seen)).filter(Boolean).join(' ');
    if (sib) return sib;
    if (p.querySelectorAll('input[type="radio"],input[type="checkbox"]').length===1) {
      const pTxt=(name(p,seen)||p.innerText||'').replace(/\s+/g,' ').trim();
      if (pTxt.length>0 && pTxt.length<=80) return pTxt;
    }
    return '';
  };
  let nameCache = new Map();
  const iconRange = document.createRange();
  const isIconEl = (el, txt) => {
    if (el.children.length > 0 || txt.length < 3 || /\s/.test(txt) || !/[a-z]/i.test(txt)) return false;
    const st = getComputedStyle(el);
    if (/icon|symbol|material/i.test(st.fontFamily.split(',')[0])) return true;
    iconRange.selectNodeContents(el);
    const w = iconRange.getBoundingClientRect().width;
    return w > 0 && w <= 1.2 * (parseFloat(st.fontSize) || 16);
  };
  const getElementByScopedId = (startEl, id) => {
    if (!id) return null;
    const root = startEl?.getRootNode ? startEl.getRootNode() : document;
    if (root?.getElementById) {
      const el = root.getElementById(id);
      if (el) return el;
    }
    const docEl = document.getElementById(id);
    if (docEl) return docEl;
    for (const r of allRoots) {
      if (r !== root && r.getElementById) {
        const el = r.getElementById(id);
        if (el) return el;
      }
    }
    return null;
  };
  const isGenericOrNumeric = s => {
    if (!s) return true;
    const trimmed = String(s).replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, '').replace(/\s+/g, ' ').trim();
    if (!trimmed) return true;
    if (/^(slider|textbox|spinbutton|searchbox|combobox|input|field)$/i.test(trimmed)) return true;
    if (/^(slider\s+)?range\s+(min(imum)?|max(imum)?)\b/i.test(trimmed)) return true;
    if (/^(leading|trailing),\s*\d+\s+to\s+\d+/i.test(trimmed)) return true;
    if (/^[$€£¥]?\s*-?\d+([.,]\d+)?\s*[kKmMbB%]?$/i.test(trimmed)) return true;
    return false;
  };
  const controlFallbackLabel = (e, seen) => {
    if (!e) return '';
    if (e.id) {
      const root = e.getRootNode ? e.getRootNode() : document;
      try {
        const lblEl = (root?.querySelector ? root : document).querySelector(`label[for="${CSS.escape(e.id)}"]`);
        if (lblEl) {
          const t = name(lblEl, seen, null, true);
          if (t && !isGenericOrNumeric(t)) return t;
        }
      } catch (_) {}
    }
    const cl = e.closest('label');
    if (cl) {
      const t = name(cl, seen, null, true);
      if (t && !isGenericOrNumeric(t)) return t;
    }
    const descAttr = e.getAttribute('aria-describedby');
    if (descAttr) {
      const descs = descAttr.split(/\s+/).map(id => {
        const el = getElementByScopedId(e, id);
        return el ? name(el, seen, null, true) : '';
      }).filter(t => t && !isGenericOrNumeric(t)).join(' ');
      if (descs) return descs;
    }
    let cur = e.parentElement;
    for (let i = 0; cur && cur !== document.body && i < 4; i++, cur = cur.parentElement) {
      if (cur.tagName === 'FIELDSET') {
        const leg = cur.querySelector('legend');
        if (leg) {
          const t = name(leg, seen, null, true);
          if (t && !isGenericOrNumeric(t)) return t;
        }
      }
      const candidates = cur.querySelectorAll('label, [class*="label-name" i], [class*="label_name" i], [class*="field-title" i], [class*="field_title" i], [class*="microcopy" i], [class*="title" i], [class*="label" i], [class*="header" i], [role="heading"], legend, h1, h2, h3, h4');
      for (const cand of candidates) {
        if (cand === e || cand.contains(e)) continue;
        const t = (cand.innerText || cand.textContent || '').replace(/\s+/g, ' ').trim();
        if (t && t.length <= 90 && !isGenericOrNumeric(t)) return t;
      }
    }
    let prev = e.previousElementSibling;
    while (prev) {
      const t = (prev.innerText || prev.textContent || '').replace(/\s+/g, ' ').trim();
      if (t && t.length <= 80 && !isGenericOrNumeric(t)) return t;
      prev = prev.previousElementSibling;
    }
    return '';
  };
  const name = (e,seen=new Set(),parentVis=null,allowHidden=false) => {
    if (!e || seen.has(e)) return '';
    const top = seen.size === 0;
    if (top && nameCache?.has(e)) return nameCache.get(e);
    seen.add(e);
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(getElementByScopedId(e,id),seen,null,true)).filter(Boolean).join(' ');
    let raw = referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen,null,true)).filter(Boolean).join(' ') ||
      choiceFallbackLabel(e,seen) ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].map(n => {
        if (n.nodeType===3) return n.textContent;
        if (n.nodeType!==1 || ['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','SVG'].includes(n.tagName)) return '';
        if (!allowHidden && (n.getAttribute('aria-hidden')==='true' || ((parentVis ??= visible(e)) && !visible(n)))) return '';
        const sub = name(n,seen,parentVis,allowHidden);
        return sub && isIconEl(n, sub) ? '' : sub;
      }).join(' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
    if (isGenericOrNumeric(raw) && (role(e) === 'slider' || e.type === 'range' || e.getAttribute('role') === 'slider')) {
      const fallback = controlFallbackLabel(e, seen);
      if (fallback) raw = fallback;
    }
    const res = raw.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, '').replace(/\s+/g, ' ').trim();
    if (top && nameCache) nameCache.set(e, res);
    return res;
  };
  const MAX_PAGE_CONTROLS = 150;
  const MAX_CONTROL_LABEL_LEN = 140;
  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'menuitemcheckbox','option','gridcell','combobox','textbox','searchbox','spinbutton',
    'slider','listbox','treeitem'];
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
      if (['text','email','url','tel','date','time','datetime-local','month','week'].includes(e.type)) return 'textbox';
      if (e.type==='range') return 'slider';
    }
    return null;
  };
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    [...allRoots.flatMap(r=>[...r.querySelectorAll('input,textarea,select')])].filter(safe)
      .map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !(visible(e) || choiceProxy(e) || choiceOverlay(e))) return null;
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
  // Native submit when the control owns a form (el.form, including form="id").
  // A role=button inside a <form> submits too, unless its type is button or reset.
  // role=link never submits. A formless control has el.form == null and no ancestor form.
  const isFormSubmitter=el=>{
    if (!el || el.getAttribute('role')==='link') return false;
    if (el.type==='button' || el.type==='reset') return false;
    const native=el.tagName==='BUTTON' || (el.tagName==='INPUT' && (el.type==='submit' || el.type==='image'));
    const roleButton=role(el)==='button';
    if (!native && !roleButton) return false;
    if (el.form!=null) return true;
    return roleButton && !!el.closest('form');
  };
  const owningForm=el=>el && (el.form || el.closest('form'));
  const hasClickHandler=el=>!!el && (el.hasAttribute('onclick') || /(^|;)\s*click:/.test(el.getAttribute('jsaction')||''));
  const looseRadioNames=new Set();
  for (const r of allRoots.flatMap(r=>[...r.querySelectorAll('input[type="radio"],[role="radio"]')])) {
    if (r.matches(':disabled') || r.closest('[aria-disabled="true"]') || owningForm(r)) continue;
    const groupName=r.getAttribute('name') || (r.closest('[role="radiogroup"]') ? 'rg'+identity(r.closest('[role="radiogroup"]')) : '');
    if (groupName) looseRadioNames.add(groupName);
  }
  // A button-like control outside any <form> commits the document's loose radio groups
  // (quiz/survey finish controls). Links, tabs, and menu items never do.
  // Native submitters and in-form role=button controls keep submits_form via isFormSubmitter.
  const isCommitControl=el=>{
    if (!el) return false;
    if (isFormSubmitter(el)) return true;
    const rname=role(el);
    if (['radio','checkbox','textbox','searchbox','combobox','option','spinbutton','link','tab','menuitem','menuitemradio','menuitemcheckbox','slider','listbox','treeitem'].includes(rname)) return false;
    if (el.tagName==='LABEL' && el.querySelector('input,textarea,select')) return false;
    const buttonLike=el.tagName==='BUTTON' || rname==='button' || hasClickHandler(el);
    if (!buttonLike || owningForm(el)) return false;
    return looseRadioNames.size>0;
  };
  const controlRequired=el=>!!el && (el.required===true || el.getAttribute('aria-required')==='true');
  const controlEmpty=el=>{
    if (!el) return false;
    if (el.tagName==='SELECT') return ![...el.selectedOptions].some(o=>String(o.value||'')!=='');
    if (el.type==='checkbox' || el.getAttribute('role')==='checkbox') return !(el.checked===true || el.getAttribute('aria-checked')==='true');
    if (el.tagName==='INPUT' || el.tagName==='TEXTAREA') return String(el.value||'').trim()==='';
    if (el.isContentEditable) return (el.innerText||'').trim()==='';
    return false;
  };
  // Custom-styled choices hide the native radio/checkbox (opacity 0) and show a styled <label>.
  // Such an input is listed through its visible label; Browser.act() clicks the label.
  const choiceProxy=e=>{
    if (e.tagName!=='INPUT' || !['radio','checkbox'].includes(e.type)) return null;
    for (const l of [...(e.labels||[]), e.closest('label')]) {
      if (!l) continue;
      const lr=l.getBoundingClientRect();
      if (lr.width>0 && lr.height>0 && visible(l)) return l;
    }
    return null;
  };
  // Invisible overlay radio/checkbox (opacity 0, stretched over its row) that takes the click at its center.
  const choiceOverlay=e=>{
    if (e.tagName!=='INPUT' || !['radio','checkbox'].includes(e.type)) return false;
    if (e.closest('[aria-hidden="true"],[inert]')) return false;
    const r=e.getBoundingClientRect();
    if (r.width<=0 || r.height<=0 || r.bottom<=0 || r.top>=innerHeight || r.right<=0 || r.left>=innerWidth) return false;
    const cx=r.left+r.width/2, cy=r.top+r.height/2;
    if (cx<0 || cx>=innerWidth || cy<0 || cy>=innerHeight) return false;
    return document.elementFromPoint(cx,cy)===e;
  };
  // Question text for repeated options ("Evet"/"Hayır"): the first line an ancestor holding the whole
  // radio group adds beyond the option labels themselves.
  const choiceQuestion=e=>{
    const group=e.name ? [...document.querySelectorAll(`input[type="${e.type}"][name="${CSS.escape(e.name)}"]`)] : [e];
    if (group.length<2) return '';
    const optionTexts=new Set(group.map(x=>(name(x)||'').trim().toLowerCase()).filter(Boolean));
    let cur=(e.closest('label')||e).parentElement;
    for (let i=0; cur && cur!==document.body && i<8; i++, cur=cur.parentElement) {
      if (!group.every(x=>cur.contains(x))) continue;
      const extra=(cur.innerText||'').split('\n').map(s=>s.replace(/\s+/g,' ').trim())
        .filter(s=>s && !optionTexts.has(s.toLowerCase()));
      if (extra.length) return extra[0].slice(0,90);
    }
    return '';
  };
  const cleanQuestion=text=>(text||'').replace(/\s*\*\s*$/,'').replace(/\s+/g,' ').trim();
  const internalGroupName=text=>/^rg\d+$/.test(cleanQuestion(text));
  const headingText=container=>{
    const head=container?.querySelector('[role="heading"],legend,h2,h3,h4');
    const text=cleanQuestion(head?.innerText);
    return internalGroupName(text) ? '' : text;
  };
  // A radiogroup often has no heading of its own. Climb to the list item, group, or
  // fieldset that asks the question, then to the repeated-option question. Never use
  // the synthetic group id (rg3) as the label.
  const groupQuestion=el=>{
    const radioGroup=el.closest('[role="radiogroup"]');
    let text=headingText(radioGroup);
    let cur=radioGroup ? radioGroup.parentElement : el.parentElement;
    for (let i=0; !text && cur && cur!==document.body && i<8; i++, cur=cur.parentElement) {
      const roleAttr=cur.getAttribute('role')||'';
      if (roleAttr==='listitem' || roleAttr==='group' || cur.tagName==='FIELDSET') text=headingText(cur);
    }
    if (!text) text=cleanQuestion(choiceQuestion(el));
    return internalGroupName(text) ? '' : text;
  };
  const computeGroupLabel = (e, rname) => {
    if (rname === 'radio') return groupQuestion(e);
    if (['checkbox', 'textbox', 'switch', 'slider'].includes(rname)) {
      const qItem = e.closest('[role="listitem"],[role="group"],[role="radiogroup"],fieldset');
      const qHead = qItem?.querySelector('[role="heading"],legend,h2,h3,h4');
      let qTxt = cleanQuestion(qHead?.innerText);
      if (!qTxt && rname !== 'textbox' && rname !== 'slider') qTxt = cleanQuestion(choiceQuestion(e));
      return internalGroupName(qTxt) ? '' : qTxt;
    }
    return '';
  };
  const computeControlLabel = (e, rname) => {
    let accessibleName = name(e) || '';
    if (isGenericOrNumeric(accessibleName) && (rname === 'slider' || e.type === 'range' || e.getAttribute('role') === 'slider')) {
      const fallback = controlFallbackLabel(e, new Set());
      if (fallback) accessibleName = fallback;
    }
    let lbl = (e.tagName === 'IFRAME' ? 'Verify you are human (Security Challenge)' : accessibleName) || rname;
    if (rname === 'link' && lbl.length > 35) {
      const hEl = e.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]');
      const hTxt = hEl?.innerText?.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, '').replace(/\s+/g, ' ').trim();
      if (hTxt && hTxt.length >= 4 && !lbl.toLowerCase().startsWith(hTxt.toLowerCase())) {
        lbl = `${hTxt} · ${lbl}`;
      }
    }
    const groupLabel = computeGroupLabel(e, rname);
    if (groupLabel && !lbl.toLowerCase().includes(groupLabel.slice(0, 25).toLowerCase())) {
      lbl = `${groupLabel.slice(0, 75)} → ${lbl.trim()}`;
    }
    return { label: lbl, accessibleName, groupLabel };
  };
  const seenCardLinks=new Set();
  const actions=[];
  for (const root of allRoots) {
    for (const e of root.querySelectorAll(selector)) {
      if (!safe(e) || e.tagName==='CANVAS' || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
      const proxy=visible(e) ? null : choiceProxy(e);
      if (!visible(e) && !proxy && !choiceOverlay(e)) continue;
      const box=proxy || e;
      const parentClickable=e.parentElement?.closest('a[href],button,[role="button"],[role="link"]');
      if (parentClickable && (name(parentClickable)||'').trim() && (name(parentClickable)||'').trim()===(name(e)||'').trim()) continue;
      const r=box.getBoundingClientRect(), rname=role(e);
      const clip=getClipRect(box.parentElement);
      const visLeft=Math.max(clip.left,r.left), visRight=Math.min(clip.right,r.right);
      const visTop=Math.max(clip.top,r.top), visBottom=Math.min(clip.bottom,r.bottom);
      const visW=visRight-visLeft, visH=visBottom-visTop;
      if (!rname || r.width<=0 || r.height<=0 || visW<Math.min(16,r.width*0.5) || visH<Math.min(20,r.height*0.35)) continue;
      if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
      const { label: lbl, accessibleName, groupLabel } = computeControlLabel(e, rname);
      const linkEl=e.tagName==='A'?e:e.closest('a[href]');
      const attrHref=linkEl?.getAttribute('href')?.trim() ?? '';
      const isPseudoHref=attrHref==='' || attrHref==='#' || /^javascript:/i.test(attrHref);
      const rawHref=isPseudoHref ? attrHref : (linkEl?.href ? linkEl.href.slice(0,240) : '');
      if (rname==='link' && rawHref && !isPseudoHref) {
        const cardScope=e.closest('article,li,tr,[class*="Card-root"],[class*="card"]') || e.parentElement;
        const scopeId=cardScope ? identity(cardScope) : 0;
        const dedupKey=`${scopeId}|${rawHref}|${lbl.trim().toLowerCase()}`;
        if (seenCardLinks.has(dedupKey)) continue;
        seenCardLinks.add(dedupKey);
      }
      const base={node:identity(e),role:rname,label:lbl,accessible_name:accessibleName,
        rect:{x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)}};
      try { e.setAttribute('data-reflex-id', String(base.node)); } catch (_) {}
      if (rawHref || rname==='link') base.href=rawHref;
      const form=owningForm(e);
      const fkey=form ? 'f'+identity(form) : null;
      if (isFormSubmitter(e) || isCommitControl(e)) {
        base.submits_form=true;
        base.form_key=fkey || 'loose';
      }
      if (rname==='radio') {
        const radioGroup=e.closest('[role="radiogroup"]');
        const groupName=e.getAttribute('name') || (radioGroup ? 'rg'+identity(radioGroup) : '');
        if (groupName) base.group=groupName;
        if (groupLabel) base.group_label=groupLabel;
        base.form_key=fkey || (groupName ? 'loose' : base.form_key);
      }
      if (controlRequired(e)) base.required=true;
      if (rname!=='radio' && controlEmpty(e)) base.empty=true;
      if (fkey && (rname==='checkbox' || e.tagName==='SELECT' || ['textbox','searchbox','spinbutton','combobox','slider'].includes(rname))) {
        base.form_key=fkey;
      } else if (!fkey && controlRequired(e) && rname!=='radio') {
        base.form_key='loose';
      }
      if (e.tagName==='SELECT' || ['textbox','searchbox','spinbutton','combobox','slider'].includes(rname)) {
        base.field_label=(name(e)||lbl).replace(/\s*\*\s*$/,'').trim();
      }
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
          (['textbox','searchbox','spinbutton','slider'].includes(rname) ||
            (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
        const value='value' in e ? String(e.value) :
          e.getAttribute('aria-valuenow') ? String(e.getAttribute('aria-valuenow')) :
          e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
        actions.push({...base,kind:editable?'fill':'click',value});
        if (editable && rname==='combobox' && !value) actions.push({...base,kind:'click',value,label:'Open '+base.label});
      }
    }
  }
  // Custom widgets (div/span answers, card toggles) carry no semantic role. Add the outermost element where
  // cursor becomes pointer, or an explicit click handler, when it is short-labelled, in view and not
  // already represented by (or wrapping) a listed control.
  const listedEls=new Set(actions.map(a=>cache.nodes.get(a.node)).filter(Boolean));
  for (const x of [...listedEls]) {
    if (x.tagName==='INPUT' && ['radio','checkbox'].includes(x.type)) {
      for (const l of [...(x.labels||[]), x.closest('label'), ...(x.parentElement?.children||[])]) {
        if (l?.tagName==='LABEL') listedEls.add(l);
      }
    }
  }
  const touchesListed=el=>{
    for (const x of listedEls) if (x===el || x.contains(el) || el.contains(x)) return true;
    return false;
  };
  const hasPointer=el=>el&&el!==document.body&&el!==document.documentElement&&getComputedStyle(el).cursor==='pointer';
  const pointerRoot=el=>{
    if (!hasPointer(el)) return false;
    if (!hasPointer(el.parentElement)) return true;
    let p=el.parentElement;
    while (p&&hasPointer(p)) {
      if (listedEls.has(p)) return false;
      p=p.parentElement;
    }
    return true;
  };
  const maxCustomArea=innerWidth*innerHeight*0.25;
  let customAdded=0;
  for (const root of allRoots) {
    if (customAdded>=60) break;
    const container = root === document ? document.body : root;
    for (const e of container.querySelectorAll('*')) {
      if (customAdded>=60) break;
      if (['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','IFRAME'].includes(e.tagName)) continue;
      const r=e.getBoundingClientRect();  // cheap filters first: most elements of a long page are off-screen
      if (r.width<=0 || r.height<=0 || r.bottom<=0 || r.top>=innerHeight || r.right<=0 || r.left>=innerWidth) continue;
      if (r.width*r.height>maxCustomArea) continue;
      if (!hasClickHandler(e) && !pointerRoot(e)) continue;
      if (e.matches(selector) || (e.closest('svg') && e.tagName.toLowerCase()!=='svg')) continue;
      if (!visible(e) || e.closest('[aria-disabled="true"]') || touchesListed(e) || e.querySelector(selector)) continue;
      const clip=getClipRect(e.parentElement);
      const visW=Math.min(clip.right,r.right)-Math.max(clip.left,r.left);
      const visH=Math.min(clip.bottom,r.bottom)-Math.max(clip.top,r.top);
      if (visW<Math.min(16,r.width*0.5) || visH<Math.min(12,r.height*0.35)) continue;
      const lbl=name(e);
      if (!lbl || lbl.length>120) continue;
      const customAction={node:identity(e),role:'button',label:lbl,accessible_name:lbl,custom:true,kind:'click',value:'',
        rect:{x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)}};
      try { e.setAttribute('data-reflex-id', String(customAction.node)); } catch (_) {}
      if (isCommitControl(e)) {
        customAction.submits_form=true;
        customAction.form_key=owningForm(e) ? 'f'+identity(owningForm(e)) : 'loose';
      }
      actions.push(customAction);
      listedEls.add(e);
      customAdded++;
    }
  }
  if (customAdded) {
    // Keep document order so System 1's first-60 element window sees custom widgets in reading order.
    const pos=a=>cache.nodes.get(a.node);
    actions.sort((a,b)=>{
      const x=pos(a), y=pos(b);
      if (!x || !y || x===y) return 0;
      return x.compareDocumentPosition(y) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
    });
  }
  const labelCounts=new Map();
  for (const a of actions) {
    if (!a.label) continue;
    const k=a.label.trim().toLowerCase();
    labelCounts.set(k,(labelCounts.get(k)||0)+1);
  }
  for (const a of actions) {
    if (!a.label) continue;
    const k=a.label.trim().toLowerCase();
    if (labelCounts.get(k)>1) {
      const el=cache.nodes.get(a.node);
      let card=el?.parentElement;
      while (card && card!==document.body && card!==document.documentElement) {
        if (card.matches('article,section,li,[role="listitem"],[role="article"],fieldset,tr,[class*="Card-root"],[class*="card" i],[class*="item" i]') &&
            card.querySelector('h1,h2,h3,h4,[role="heading"],legend,[class*="title" i]')) {
          break;
        }
        card=card.parentElement;
      }
      if (!card) {
        let cur=el?.parentElement;
        while (cur && cur!==document.body && cur!==document.documentElement) {
          const h=cur.querySelector('h1,h2,h3,h4,[role="heading"],legend');
          if (h && !h.contains(el)) { card=cur; break; }
          cur=cur.parentElement;
        }
      }
      const hEl=card?.querySelector('h1,h2,h3,h4,[role="heading"],legend,[class*="title" i]');
      const hTxt=(hEl?.innerText||hEl?.textContent||'').replace(/\s+/g,' ').trim();
      if (hTxt && !a.label.toLowerCase().includes(hTxt.slice(0,25).toLowerCase())) {
        a.label=`${a.label.trim()} (${hTxt.slice(0,90)})`;
      }
    }
  }
  // Form fields that still share a label (min/max boxes under one heading) get their own field
  // caption and placeholder, so "BATHROOMS · min" is not mistaken for the bedroom box.
  const stillCounts=new Map();
  for (const a of actions) {
    if (a.label) stillCounts.set(a.label.trim().toLowerCase(),(stillCounts.get(a.label.trim().toLowerCase())||0)+1);
  }
  for (const a of actions) {
    if (!a.label || stillCounts.get(a.label.trim().toLowerCase())<2) continue;
    const el=cache.nodes.get(a.node);
    if (!el || !['INPUT','TEXTAREA','SELECT'].includes(el.tagName)) continue;
    const low=a.label.toLowerCase(), parts=[];
    const cap=controlFallbackLabel(el,new Set());
    if (cap && !low.includes(cap.slice(0,25).toLowerCase())) parts.push(cap.slice(0,60));
    const ph=(el.getAttribute('placeholder')||'').replace(/\s+/g,' ').trim();
    if (ph && !low.includes(ph.toLowerCase()) && !parts.some(p=>p.toLowerCase().includes(ph.toLowerCase()))) parts.push(ph.slice(0,40));
    if (parts.length) a.label=`${parts.join(' · ')} · ${a.label.trim()}`;
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
  const visuallyRendered = e => {
    if (!e || e.closest('[inert]')) return false;
    if (e.checkVisibility) {
      return e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
    }
    const st = window.getComputedStyle(e);
    return st.display !== 'none' && st.visibility !== 'hidden' && parseFloat(st.opacity || '1') > 0;
  };
  const words=[], fullWords=[];
  const range=document.createRange(); let length=0,fullLength=0;
  for (const root of allRoots) {
    if (length>=6000 && fullLength>=30000) break;
    const container = root === document ? document.body : root;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let node;
    while ((node=walker.nextNode()) && (length<6000 || fullLength<30000)) {
      const value=node.textContent.trim(), parent=node.parentElement;
      if (!value || !parent || parent.closest('script,style,noscript,template') || !visuallyRendered(parent)) continue;
      range.selectNodeContents(node); const r=range.getBoundingClientRect();
      if (r.width<=0 || r.height<=0 || r.top>=getClipBottom(parent)-2) continue;
      if (fullLength<30000) {
        fullWords.push(value); fullLength+=value.length;
      }
      if (length<6000 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
        words.push(value); length+=value.length;
      }
    }
  }
  const isUnclipped=el=>{
    if (!visuallyRendered(el)) return false;
    const r=el.getBoundingClientRect();
    return r.width>0 && r.height>0 && r.top<getClipBottom(el.parentElement)-2;
  };
  const h12=[...allRoots.flatMap(r=>[...r.querySelectorAll('h1,h2')])].filter(isUnclipped)
    .map(h=>`[${h.tagName.toLowerCase()}] `+h.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
  const h34=[...allRoots.flatMap(r=>[...r.querySelectorAll('h3,h4')])].filter(isUnclipped)
    .map(h=>`[${h.tagName.toLowerCase()}] `+h.innerText.replace(/\s+/g,' ').trim()).filter(Boolean);
  const headings=[...new Set([...h12,...h34])].slice(0,80);
  const seenPageHrefs=new Set(actions.map(a=>a.href).filter(Boolean));
  const pageLinksByHref=new Map();
  for (const a of allRoots.flatMap(r=>[...r.querySelectorAll('a[href]')])) {
    const attrHref=a.getAttribute('href')?.trim() ?? '';
    if (!attrHref || attrHref==='#' || /^javascript:|^mailto:|^tel:/i.test(attrHref)) continue;
    const rawHref=a.href ? a.href.slice(0,240) : '';
    if (!rawHref || seenPageHrefs.has(rawHref) || !isUnclipped(a)) continue;
    const lbl=name(a).slice(0,140);
    const prev=pageLinksByHref.get(rawHref);
    if (!prev) {
      if (pageLinksByHref.size<150) pageLinksByHref.set(rawHref,{role:'link',label:lbl||rawHref,href:rawHref});
    } else if (lbl && prev.label===rawHref) {
      prev.label=lbl;
    }
  }
  const page_links=[...pageLinksByHref.values()];
  const page_controls = [];
  const seenActionNodes = new Set(actions.map(a => a.node).filter(n => typeof n === 'number'));
  const seenControlNodes = new Set();
  const formControlSel = 'input,select,textarea,[role="radio"],[role="checkbox"],[role="option"],[role="switch"],[role="slider"],[role="listbox"],[role="menuitemcheckbox"],[role="treeitem"]';
  const FORM_CONTROL_ROLES = new Set(['radio', 'checkbox', 'switch', 'option', 'combobox', 'textbox', 'searchbox', 'spinbutton', 'slider', 'listbox', 'menuitemcheckbox', 'treeitem']);

  for (const e of allRoots.flatMap(r=>[...r.querySelectorAll(formControlSel)])) {
    if (page_controls.length >= MAX_PAGE_CONTROLS) break;
    if (!safe(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]')) continue;
    const rname = role(e);
    if (!rname || !FORM_CONTROL_ROLES.has(rname)) continue;

    const proxy = visible(e) ? null : choiceProxy(e);
    const box = proxy || e;
    if (!isUnclipped(box)) continue;

    if (cache.ids.has(e) && seenActionNodes.has(cache.ids.get(e))) continue;

    const nodeId = identity(e);
    if (seenControlNodes.has(nodeId)) continue;
    seenControlNodes.add(nodeId);

    let { label: ctrlLabel } = computeControlLabel(e, rname);
    if (ctrlLabel.length > MAX_CONTROL_LABEL_LEN) {
      ctrlLabel = ctrlLabel.slice(0, MAX_CONTROL_LABEL_LEN);
    }

    const kind = ['textbox', 'searchbox', 'spinbutton', 'slider'].includes(rname) ||
                 (['INPUT', 'TEXTAREA'].includes(e.tagName) && !['checkbox', 'radio'].includes(e.type)) ? 'fill' :
                 (e.tagName === 'SELECT' ? 'select' : 'click');

    const boxRect = box.getBoundingClientRect();
    const ctrlObj = {
      node: nodeId,
      role: rname,
      kind,
      label: ctrlLabel,
      // Lets System 2 know a filled field is still there after scrolling past it.
      where: boxRect.bottom <= 0 ? 'above' : (boxRect.top >= innerHeight ? 'below' : 'in view'),
    };

    if (['checkbox', 'radio'].includes(e.type)) {
      ctrlObj.checked = String(e.checked);
    } else if (e.hasAttribute('aria-checked')) {
      ctrlObj.checked = e.getAttribute('aria-checked');
    }

    if (e.hasAttribute('aria-selected')) {
      ctrlObj.selected = e.getAttribute('aria-selected');
    } else if ('selected' in e && e.selected !== undefined) {
      ctrlObj.selected = String(e.selected);
    }

    if (e.tagName === 'SELECT') {
      const selVal = [...e.selectedOptions].map(o => o.label || o.value).join(', ');
      if (selVal) ctrlObj.value = selVal.slice(0, 80);
    } else if ('value' in e && e.value) {
      ctrlObj.value = String(e.value).slice(0, 80);
    } else if (e.hasAttribute('aria-valuenow')) {
      ctrlObj.value = String(e.getAttribute('aria-valuenow')).slice(0, 80);
    }

    page_controls.push(ctrlObj);
  }
  const dialogs=Array.isArray(window.__reflexDialogs)?window.__reflexDialogs.slice(0,5):[];
  const dlgPrefix=dialogs.length?dialogs.join('\n')+'\n':'';
  const text=(dlgPrefix+words.join('\n')).slice(0,6000), full_text=(dlgPrefix+fullWords.join('\n')).slice(0,30000), height=document.documentElement.scrollHeight;
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  for (const c of page_controls) if (!(c.node in guards)) guards[c.node]=cache.guard(cache.nodes.get(c.node));
  nameCache=null;
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6]];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  if (scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:420});
  if (scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-420});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,full_text,headings,page_links,page_controls,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions,has_iframes:!!document.querySelector('iframe')};
})()
