/* Target-voice covers reuse the workbench's connection, queue and audio player. */
(() => {
  'use strict';
  const nav = el('button', {id:'tab-covers', type:'button', role:'tab', 'data-tab':'covers',
    'aria-selected':'false', 'aria-controls':'view-covers'}, [el('span',{class:'nav-number',text:'06'}), document.createTextNode('目标音色翻唱')]);
  document.querySelector('.nav').append(nav);
  const view = el('section',{id:'view-covers',role:'tabpanel','aria-labelledby':'tab-covers',hidden:''});
  view.innerHTML = `
    <div class="page-head"><div><p class="eyebrow">A VOICE, A NEW PERFORMANCE</p><h1>让旋律，遇见你的声音。</h1><p>用已训练的目标音色转换演唱干声，保留原来的旋律与节奏。</p></div><div class="edition"><b>声 / 06</b>LOCAL VOICE CONVERSION</div></div>
    <div class="training-steps"><div class="training-step"><b>01 · 准备音色</b><p>导入 RVC 模型，或用干净录音训练自己的音色。</p></div><div class="training-step"><b>02 · 输入演唱</b><p>上传不含伴奏的演唱人声，按需要设置升降调。</p></div><div class="training-step"><b>03 · 生成与收藏</b><p>试听转换后的干声，可选混入独立伴奏，结果自动收录。</p></div></div>
    <section class="panel"><div class="panel-head"><h2 class="panel-title">转换与训练组件</h2><span id="cover-assets-badge" class="badge">尚未检查</span></div><div class="panel-body"><p id="cover-assets-info" class="hint">首次使用需要下载本地组件。</p><div class="button-row" style="margin-top:14px"><button id="cover-download" type="button" class="btn btn-small">下载组件</button><button id="cover-refresh" type="button" class="btn btn-small">检查状态</button></div></div></section>
    <div class="layout" style="margin-top:22px"><div>
      <form id="cover-form" class="panel"><div class="panel-head"><h2 class="panel-title">生成目标音色翻唱</h2><small>输入为演唱人声</small></div><div class="panel-body">
        <label class="field"><span class="field-label">目标音色</span><select id="cover-model" required><option value="">请先导入或训练音色</option></select></label>
        <p id="cover-voice-note" class="hint" role="status" style="margin-top:-10px;margin-bottom:18px"></p>
        <label class="field"><span class="field-label">演唱干声文件 <span>0.5–600 秒 / ≤150MB</span></span><input id="cover-source-file" type="file" accept="audio/*"></label>
        <div class="button-row" style="margin-bottom:18px"><button id="cover-load-test" type="button" class="btn btn-small">载入测试干声</button></div>
        <label class="field"><span class="field-label">或使用本机人声路径</span><input id="cover-source" required class="mono" placeholder="选择文件后会自动填入，也可填写本机路径"></label>
        <p class="hint">干声提供旋律和歌词，目标音色模型决定音色；训练音色使用自己的录音。</p>
        <p class="hint">请提供独立人声。含伴奏的整首歌曲需先分离人声，本页不会自动分离。</p>
        <div class="form-grid" style="margin-top:18px"><label class="field"><span class="field-label">升降调 <span>半音</span></span><input id="cover-pitch" type="number" min="-24" max="24" step="1" value="0"></label><label class="field"><span class="field-label">索引混合比例 <span>无 .index 时不启用</span></span><input id="cover-index-rate" type="number" min="0" max="1" step="0.05" value="0.5"></label></div>
        <details><summary>伴奏与高级设置</summary><div style="margin-top:16px"><label class="field"><span class="field-label">独立伴奏文件 <span>可选</span></span><input id="cover-accompaniment-file" type="file" accept="audio/*"></label><label class="field"><span class="field-label">伴奏路径 <span>留空则只输出干声</span></span><input id="cover-accompaniment" class="mono"></label><div class="form-grid"><label class="field"><span class="field-label">人声音量</span><input id="cover-vocal-gain" type="number" min="0" max="2" step="0.1" value="1"></label><label class="field"><span class="field-label">伴奏音量</span><input id="cover-accompaniment-gain" type="number" min="0" max="2" step="0.1" value="0.7"></label><label class="field"><span class="field-label">模型内音色编号</span><input id="cover-speaker" type="number" min="0" max="10000" value="0"></label><label class="field"><span class="field-label">随机种子</span><input id="cover-seed" type="number" min="0" max="2147483647" value="42"></label></div></div></details>
        <button id="cover-submit" type="submit" class="btn btn-primary btn-wide" style="margin-top:20px">生成翻唱 →</button><p id="cover-message" class="hint" role="status"></p>
      </div></form>
      <section class="panel" id="cover-result"><div class="panel-head"><h2 class="panel-title">翻唱与训练任务</h2></div><div class="panel-body"><div id="cover-job-list" class="button-row"></div><div id="cover-job-detail" style="margin-top:16px"><p class="hint">提交后在这里查看进度、日志和音频。</p></div></div></section>
    </div><div>
      <form id="cover-register-form" class="panel"><div class="panel-head"><h2 class="panel-title">导入已有 RVC 音色</h2></div><div class="panel-body"><label class="field"><span class="field-label">音色名称</span><input id="cover-register-name" required maxlength="100"></label><label class="field"><span class="field-label">模型 .pth 路径</span><input id="cover-register-checkpoint" class="mono" required></label><label class="field"><span class="field-label">检索 .index 路径 <span>可选</span></span><input id="cover-register-index" class="mono"></label><button id="cover-register" class="btn btn-dark btn-wide" type="submit">导入音色</button><p class="hint">支持 RVC v2、带 F0 的本地模型。短参考录音不会自动变成 RVC 音色模型。</p></div></form>
      <form id="cover-training-form" class="panel"><div class="panel-head"><h2 class="panel-title">训练目标音色</h2><small>RVC v2 / 40kHz</small></div><div class="panel-body"><label class="field"><span class="field-label">训练后音色名称</span><input id="cover-training-name" required maxlength="100" value="我的翻唱音色"></label><label class="field"><span class="field-label">干净 WAV 录音目录</span><input id="cover-training-directory" required class="mono" placeholder="本机目录，无需 CSV 或逐句转写"></label><p class="hint">使用同一人的清晰录音；歌唱录音有助于覆盖音域与长音。少量数据只能检查流程，不能保证音色质量。</p><div class="form-grid" style="margin-top:16px"><label class="field"><span class="field-label">训练轮数</span><input id="cover-training-epochs" type="number" min="1" max="10000" value="20"></label><label class="field"><span class="field-label">批大小</span><input id="cover-training-batch" type="number" min="1" max="16" value="1"></label><label class="field"><span class="field-label">学习率</span><input id="cover-training-lr" type="number" min="0.000001" max="0.001" step="0.000001" value="0.0001"></label><label class="field"><span class="field-label">最多更新步数 <span>可选</span></span><input id="cover-training-steps" type="number" min="1" max="1000000" placeholder="不限制"></label></div><div class="button-row"><button id="cover-validate" class="btn" type="button">检查录音</button><button id="cover-train" class="btn btn-dark" type="submit">开始训练</button></div><div id="cover-training-report" style="margin-top:15px"></div><p class="hint">训练成功后自动导入音色。CPU 训练速度较慢；可在“模型与任务”查看设备与日志。</p></div></form>
    </div></div>`;
  document.querySelector('.content').append(view);
  let data = null, selectedJob = '', signature = '', statusLoading = false;
  const message = (text, bad=false) => { $('cover-message').textContent=text; $('cover-message').className='hint '+(bad?'error':'success'); };
  const demoVoice = model => model.demo === true || (model.demo !== false && model.name === 'RVC 合成样本验收音色');
  function renderVoiceNote() {
    const model = data && data.voices.find(voice => voice.id === $('cover-model').value);
    const note = $('cover-voice-note');
    note.textContent = model ? model.quality_note || (demoVoice(model) ? '此音色仅用合成样本完成一次更新，用于验证转换流程，不代表可用的音色质量。请导入已训练好的音色，或使用自己的录音充分训练。' : '') : '';
    note.className = 'hint' + (model && demoVoice(model) ? ' error' : '');
  }
  async function refresh() {
    if (statusLoading) return;
    statusLoading=true;
    try {
      data=await api('/api/covers/status');
      const ready=Boolean(data.assets.ready);
      $('cover-assets-badge').textContent=ready?'组件已就绪':'需要下载组件';
      $('cover-assets-badge').className='badge '+(ready?'green':'');
      $('cover-assets-info').textContent=ready?'音频编码器、音高模型和训练基础权重已保存在本机。':'首次下载约 775MB。下载成功后，转换与训练只读取本地文件。';
      $('cover-download').hidden=ready;
      const select=$('cover-model'), previous=select.value;
      select.replaceChildren();
      for (const model of data.voices) select.append(el('option',{value:model.id,text:model.name+(demoVoice(model)?' · 仅流程测试':'')+(model.ready?'':' · 未就绪')}));
      if (!data.voices.length) select.append(el('option',{value:'',text:'请先导入或训练目标音色'}));
      if (data.voices.some(m=>m.id===previous)) select.value=previous;
      renderVoiceNote();
      $('cover-submit').disabled=!data.voices.some(m=>m.ready);
      sync();
    } catch(error) { message(error.message,true); }
    finally { statusLoading=false; }
  }
  function sync() {
    if (view.hidden) return;
    const relevant=jobs.filter(j=>['cover','train_voice'].includes(j.kind)||j.model_id==='rvc-support');
    if (!selectedJob && relevant.length) selectedJob=relevant[0].id;
    const list=$('cover-job-list'); list.replaceChildren();
    for (const job of relevant.slice(0,6)) list.append(el('button',{type:'button',class:'btn btn-small',text:jobLabel(job)+' · '+(jobStates[job.status]||job.status),onClick:()=>{selectedJob=job.id;signature='';sync();}}));
    const job=relevant.find(j=>j.id===selectedJob);
    if (!job) return;
    const next=jobSignature(job); if(next===signature) return;
    signature=next;
    const host=$('cover-job-detail');host.replaceChildren();appendJobContent(host,job,true);
    if (job.result && job.result.vocal_audio_url) { host.append(el('p',{class:'hint',text:'转换后的独立干声'}));appendAudio(host,{...job.result,audio_url:job.result.vocal_audio_url},job.id+'-vocal'); }
  }
  async function submitted(job) {selectedJob=job.id;signature='';message('任务已提交，可在下方查看进度。');await refreshAll();sync();}
  nav.addEventListener('click',()=>{switchTab('covers');refresh();});
  $('cover-model').addEventListener('change',renderVoiceNote);
  $('cover-load-test').addEventListener('click',()=>busy($('cover-load-test'),'正在载入…',async()=>{
    const sample = await api('/api/covers/test-vocal');
    $('cover-source').value = sample.path;
    $('cover-source-file').value = '';
    message('已载入无伴奏测试歌声，' + sample.name + '（' + Number(sample.duration_seconds).toFixed(1) + ' 秒）；用于转换测试。');
  }));
  $('cover-refresh').addEventListener('click',()=>busy($('cover-refresh'),'正在检查…',refresh));
  $('cover-download').addEventListener('click',()=>busy($('cover-download'),'正在提交…',async()=>submitted(await api('/api/covers/assets',{method:'POST'}))));
  for (const [input, path, kind] of [['cover-source-file','cover-source','source'],['cover-accompaniment-file','cover-accompaniment','accompaniment']]) {
    $(input).addEventListener('change',async()=>{
      const file=$(input).files[0];if(!file)return;
      $(input).disabled=true;message('正在上传并检查音频…');
      try { const form=new FormData();form.append('file',file);const result=await api('/api/covers/audio?kind='+kind,{method:'POST',body:form});$(path).value=result.path;message('音频已准备好，时长 '+Number(result.duration_seconds).toFixed(1)+' 秒。'); }
      catch(error) {message(error.message,true);notify(error.message,true);}
      finally {$(input).disabled=false;}
    });
  }
  $('cover-form').addEventListener('submit',event=>{event.preventDefault();busy($('cover-submit'),'正在提交…',async()=>{
    const body={model_id:$('cover-model').value,source_audio:$('cover-source').value,pitch_shift:Number($('cover-pitch').value),index_rate:Number($('cover-index-rate').value),speaker_id:Number($('cover-speaker').value),seed:Number($('cover-seed').value),accompaniment_audio:$('cover-accompaniment').value,vocal_gain:Number($('cover-vocal-gain').value),accompaniment_gain:Number($('cover-accompaniment-gain').value)};
    await submitted(await api('/api/covers/convert',{method:'POST',body}));
  });});
  $('cover-register-form').addEventListener('submit',event=>{event.preventDefault();busy($('cover-register'),'正在导入…',async()=>{
    const model=await api('/api/covers/models',{method:'POST',body:{name:$('cover-register-name').value,checkpoint_path:$('cover-register-checkpoint').value,index_path:$('cover-register-index').value}});
    await refresh();$('cover-model').value=model.id;renderVoiceNote();await refreshAll();notify('音色已导入。');
  });});
  async function validate() {
    const report=await api('/api/covers/datasets/validate',{method:'POST',body:{wavs_dir:$('cover-training-directory').value}});
    const host=$('cover-training-report');host.replaceChildren(el('p',{class:'hint '+(report.valid?'success':'error'),text:report.valid?'录音检查通过：'+report.sample_count+' 段，共 '+(report.total_duration_seconds/60).toFixed(1)+' 分钟。':'录音检查未通过。'}));
    for (const text of [...report.errors,...report.warnings]) host.append(el('p',{class:'hint',text}));
    return report;
  }
  $('cover-validate').addEventListener('click',()=>busy($('cover-validate'),'正在检查…',validate));
  $('cover-training-form').addEventListener('submit',event=>{event.preventDefault();busy($('cover-train'),'正在提交…',async()=>{
    const body={name:$('cover-training-name').value,wavs_dir:$('cover-training-directory').value,epochs:Number($('cover-training-epochs').value),batch_size:Number($('cover-training-batch').value),learning_rate:Number($('cover-training-lr').value)};
    if($('cover-training-steps').value)body.max_steps=Number($('cover-training-steps').value);
    await submitted(await api('/api/covers/train',{method:'POST',body}));
  });});
  window.StudioCovers={refresh,sync};
})();
