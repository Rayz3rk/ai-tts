/* Original songs use ACE-Step's own composer; no manual score is required. */
(() => {
  'use strict';
  const nav=el('button',{id:'tab-music',type:'button',role:'tab','data-tab':'music','aria-selected':'false','aria-controls':'view-music'},[el('span',{class:'nav-number',text:'08'}),document.createTextNode('歌词自动创作')]);
  document.querySelector('.nav').append(nav);
  const view=el('section',{id:'view-music',role:'tabpanel','aria-labelledby':'tab-music',hidden:''});
  view.innerHTML=`<div class="page-head"><div><p class="eyebrow">A SONG STARTS WITH YOUR WORDS</p><h1>写下歌词，听见新歌。</h1><p>描述你想要的风格，让本地模型创作旋律、演唱和伴奏。</p></div><div class="edition"><b>曲 / 08</b>ORIGINAL SONG</div></div>
    <section class="panel"><div class="panel-head"><h2 class="panel-title">ACE-Step 1.5</h2><span id="music-ready" class="badge">尚未检查</span></div><div class="panel-body"><p id="music-info" class="hint">自动创作模型与独立显卡运行环境。</p><div class="button-row" style="margin-top:14px"><button id="music-refresh" type="button" class="btn btn-small">检查状态</button><button id="music-download" type="button" class="btn btn-small">下载创作模型</button></div></div></section>
    <div class="layout" style="margin-top:22px"><div><form id="music-form" class="panel"><div class="panel-head"><h2 class="panel-title">创作一首歌</h2><button id="music-example" type="button" class="btn btn-small">载入示例</button></div><div class="panel-body">
    <label class="field"><span class="field-label">作品名称</span><input id="music-title" required maxlength="100" value="晨光里的小路"></label>
    <label class="field"><span class="field-label">歌词 <span>可用 [Verse]、[Chorus] 划分主歌和副歌</span></span><textarea id="music-lyrics" required rows="10" maxlength="4000"></textarea></label>
    <label class="field"><span class="field-label">音乐风格</span><textarea id="music-style" required rows="3" maxlength="1000" placeholder="例如：温暖的普通话流行歌曲，女声，钢琴与木吉他，轻快节奏"></textarea></label>
    <button id="music-submit" type="submit" class="btn btn-primary btn-wide">生成原创歌曲 →</button><p id="music-message" class="hint" role="status"></p></div></form>
    <section class="panel"><div class="panel-head"><h2 class="panel-title">歌曲与任务</h2></div><div class="panel-body"><div id="music-jobs" class="button-row"></div><div id="music-result" style="margin-top:16px"><p class="hint">生成后可试听、下载，歌曲会自动收录到音频素材库。</p></div></div></section></div>
    <div><section class="panel"><div class="panel-head"><h2 class="panel-title">歌曲设置</h2></div><div class="panel-body">
    <label class="field"><span class="field-label">时长（秒） <span>首次建议 30 秒试听</span></span><input id="music-duration" required type="number" min="10" max="180" value="30"></label>
    <label class="field"><span class="field-label">歌词语言</span><select id="music-language"><option value="zh">普通话</option><option value="en">英语</option><option value="ja">日语</option><option value="ko">韩语</option><option value="unknown">自动判断</option></select></label>
    <label class="field"><span class="field-label">速度 BPM <span>留空由模型创作</span></span><input id="music-bpm" type="number" min="30" max="300" placeholder="自动"></label>
    <details><summary>随机种子</summary><label class="field" style="margin-top:15px"><span class="field-label">改变种子可尝试另一种旋律</span><input id="music-seed" type="number" min="0" max="2147483647" value="42" required></label></details></div></section>
    <section class="panel"><div class="panel-body"><h2 class="panel-title">选择适合的创作方式</h2><p class="hint">这里会创作新的旋律和伴奏，演唱音色由生成模型决定。歌词遵循度、曲风和人声质量可能随种子变化。</p><p class="hint">已有旋律要精确填词，请使用“歌词与旋律演唱”；已有演唱要换成个人音色，请使用“目标音色翻唱”，输入独立人声。</p><p class="hint">本机采用 2B turbo 与 0.6B 音乐语言模型，显卡本地运行。长歌词请配合更长时长或分段生成。</p></div></section></div></div>`;
  document.querySelector('.content').append(view);
  let loading=false,selected='',signature='';
  function example(){ $('music-lyrics').value='[Verse]\n晨光落在安静的小路\n微风带着花开的温度\n我把昨天轻轻地放下\n向着远方迈开了脚步\n\n[Chorus]\n让音乐带来好心情\n让每个明天都有回音';$('music-style').value='Mandarin pop, warm clear female vocal, piano and acoustic guitar, bright gentle drums, uplifting melody'; }
  async function refresh(){if(loading)return;loading=true;try{const data=await api('/api/music/status');const ready=data.model.ready;$('music-ready').textContent=ready?'创作模型已就绪':'需要准备模型';$('music-ready').className='badge'+(ready?' green':'');$('music-info').textContent=ready?'已在本机准备好独立显卡运行环境与原创歌曲模型。':'首次需约 7.2GB 权重；独立环境需使用项目的自动创作安装脚本准备。';$('music-submit').disabled=!ready;$('music-download').hidden=ready;sync();}catch(error){notify(error.message,true);}finally{loading=false;}}
  function sync(){if(view.hidden)return;const relevant=jobs.filter(j=>j.kind==='compose'||j.model_id==='ace-step-15');if(!selected&&relevant.length)selected=relevant[0].id;const list=$('music-jobs');list.replaceChildren();for(const job of relevant.slice(0,6))list.append(el('button',{type:'button',class:'btn btn-small',text:jobLabel(job)+' · '+(jobStates[job.status]||job.status),onClick:()=>{selected=job.id;signature='';sync();}}));const job=relevant.find(j=>j.id===selected);if(!job)return;const next=jobSignature(job);if(next===signature)return;signature=next;$('music-result').replaceChildren();appendJobContent($('music-result'),job,false);if(job.result&&job.result.attribution)$('music-result').append(el('p',{class:'hint',text:job.result.attribution+' · 包含演唱与伴奏'}));}
  async function submitted(job){selected=job.id;signature='';await refreshAll();sync();notify('任务已提交，可在下方查看进度。');}
  nav.addEventListener('click',()=>{switchTab('music');refresh();});
  $('music-refresh').addEventListener('click',()=>busy($('music-refresh'),'正在检查…',refresh));
  $('music-download').addEventListener('click',()=>busy($('music-download'),'正在提交…',async()=>submitted(await api('/api/music/assets',{method:'POST'}))));
  $('music-example').addEventListener('click',example);
  $('music-form').addEventListener('submit',event=>{event.preventDefault();busy($('music-submit'),'正在提交…',async()=>{const body={title:$('music-title').value,lyrics:$('music-lyrics').value,style:$('music-style').value,duration:Number($('music-duration').value),language:$('music-language').value,seed:Number($('music-seed').value)};if($('music-bpm').value)body.bpm=Number($('music-bpm').value);await submitted(await api('/api/music/generate',{method:'POST',body}));});});
  jobKinds.compose='歌词自动创作';window.StudioMusic={refresh,sync};example();
})();
