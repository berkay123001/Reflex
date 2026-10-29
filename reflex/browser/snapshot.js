(() => {
  if (!document.body) return null;
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
  const isIconEl = (el, txt, hasOtherTxt) => {
    if (el.children.length > 0 || txt.length < 3 || /\s/.test(txt) || !/[a-z]/i.test(txt)) return false;
    if (hasOtherTxt || el.tagName === 'IMG' || txt.length >= 5) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0 && r.width <= 1.5 * r.height) return true;
    }
    return /icon|symbol|material/i.test(getComputedStyle(el).fontFamily.split(',')[0]);
  };
  const name = (e,seen=new Set(),parentVis=null) => {
    if (!e || seen.has(e)) return '';
    const top = seen.size === 0;
    if (top && nameCache?.has(e)) return nameCache.get(e);
    seen.add(e);
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(document.getElementById(id),seen)).filter(Boolean).join(' ');
    const raw = referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      choiceFallbackLabel(e,seen) ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : (() => {
        const parts=[];
        for (const n of e.childNodes) {
          if (n.nodeType===3) {
            if (n.textContent.trim()) parts.push({el:null,txt:n.textContent});
          } else if (n.nodeType===1 && !['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','SVG'].includes(n.tagName) && n.getAttribute('aria-hidden')!=='true' && (!(parentVis ??= visible(e)) || visible(n))) {
            const sub=name(n,seen,parentVis);
            if (sub) parts.push({el:n,txt:sub});
          }
        }
        const multi=parts.length>1;
        return parts.map(p=>p.el && isIconEl(p.el,p.txt,multi) ? '' : p.txt).join(' ').trim();
      })()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
    const res = raw.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g, '').replace(/\s+/g, ' ').trim();
    if (top && nameCache) nameCache.set(e, res);
    return res;
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
  for (const r of document.querySelectorAll('input[type="radio"],[role="radio"]')) {
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
    if (['radio','checkbox','textbox','searchbox','combobox','option','spinbutton','link','tab','menuitem','menuitemradio'].includes(rname)) return false;
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
  const seenCardLinks=new Set();
  const actions=[];
  for (const e of document.querySelectorAll(selector)) {
    if (!safe(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
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
    const accessibleName=name(e)||'';
    let lbl=(e.tagName==='IFRAME'?'Verify you are human (Security Challenge)':accessibleName)||rname;
    if (rname==='link' && lbl.length>35) {
      const hEl=e.querySelector('h1,h2,h3,h4,[class*="title"],[class*="Title"]');
      const hTxt=hEl?.innerText?.replace(/[\u200b-\u200f\u202a-\u202e\ufeff]/g,'').replace(/\s+/g,' ').trim();
      if (hTxt && hTxt.length>=4 && !lbl.toLowerCase().startsWith(hTxt.toLowerCase())) {
        lbl=`${hTxt} · ${lbl}`;
      }
    }
    let groupLabel='';
    if (rname==='radio') {
      groupLabel=groupQuestion(e);
    } else if (['checkbox','textbox'].includes(rname)) {
      const qItem=e.closest('[role="listitem"],[role="group"],[role="radiogroup"],fieldset');
      const qHead=qItem?.querySelector('[role="heading"],legend,h2,h3,h4');
      let qTxt=cleanQuestion(qHead?.innerText);
      if (!qTxt && rname!=='textbox') qTxt=cleanQuestion(choiceQuestion(e));
      groupLabel=internalGroupName(qTxt) ? '' : qTxt;
    }
    if (groupLabel && !lbl.toLowerCase().includes(groupLabel.slice(0,25).toLowerCase())) {
      lbl=`${groupLabel.slice(0,75)} → ${lbl.trim()}`;
    }
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
    if (fkey && (rname==='checkbox' || e.tagName==='SELECT' || ['textbox','searchbox','spinbutton','combobox'].includes(rname))) {
      base.form_key=fkey;
    } else if (!fkey && controlRequired(e) && rname!=='radio') {
      base.form_key='loose';
    }
    if (e.tagName==='SELECT' || ['textbox','searchbox','spinbutton','combobox'].includes(rname)) {
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
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      if (editable && rname==='combobox' && !value) actions.push({...base,kind:'click',value,label:'Open '+base.label});
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
  const pointerRoot=el=>getComputedStyle(el).cursor==='pointer' &&
    (!el.parentElement || getComputedStyle(el.parentElement).cursor!=='pointer');
  const maxCustomArea=innerWidth*innerHeight*0.25;
  let customAdded=0;
  for (const e of document.body.querySelectorAll('*')) {
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
    if (isCommitControl(e)) {
      customAction.submits_form=true;
      customAction.form_key=owningForm(e) ? 'f'+identity(owningForm(e)) : 'loose';
    }
    actions.push(customAction);
    listedEls.add(e);
    customAdded++;
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
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,full_text,headings,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
})()
