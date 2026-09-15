
    (()=>{
      const root=document.getElementById('op-general-bench'),q=s=>root.querySelector(s);
      const appearance={spacing:12};
      const uid=()=>globalThis.crypto?.randomUUID?crypto.randomUUID():'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.floor(Math.random()*16);return(c==='x'?r:(r&3|8)).toString(16);});
      const clone=x=>JSON.parse(JSON.stringify(x));
      const gaussian=(n=1,w=532e-9)=>({type:'IncidentSource',id:uid(),name:'Gaussian source',kind:'gaussian',grid:{nx:128,ny:128,dx:1e-6,dy:1e-6},wavelength_m:w,medium_index:n,components:['Ex','Ey'],amplitudes:[1,.5],waist_x_m:10e-6,waist_y_m:10e-6});
      const blank=()=>({format:'optiprop-project',schema_version:1,project_id:uid(),name:'未命名專案',source:gaussian(),optical_system:{type:'OpticalSystem',id:uid(),name:'Optical system',layers:[]},compute_settings:{device:'cpu',precision:'complex128'},assets:[],metadata:{}});
      const prop=(d=100e-6,n=1)=>({type:'PropagationLayer',id:uid(),name:'傳遞段',spec:{method:'asm',distance_m:d,medium_index:n,padding:{mode:'factor',factor:2}},keep_intermediate:true});
      const element=(type,n)=>type==='IdealLensLayer'?{type,id:uid(),name:'理想透鏡',focal_length_m:1e-3,phase_model:'exact_equal_path'}:type==='ApertureLayer'?{type,id:uid(),name:'圓孔徑',aperture:{shape:'circle',size_x_m:50e-6}}:{type,id:uid(),name:'介質界面',n1:n,n2:n===1?1.5:1,ignore_reflection:false};
      const kinds={IncidentSource:'光源',PropagationLayer:'傳遞',IdealLensLayer:'透鏡 · 理想相位',Binary2LensLayer:'透鏡 · Binary2',ApertureLayer:'孔徑',InterfaceLayer:'介面'};
      const modelFields={IdealLensLayer:['focal_length_m','phase_model','design_wavelength_m','medium_index'],Binary2LensLayer:['coefficients']};
      const modelDefaults={IdealLensLayer:{focal_length_m:1e-3,phase_model:'exact_equal_path'},Binary2LensLayer:{coefficients:[0,0,0,0]}};
      let project=blank(),selected=project.source.id,history=[],examples={},currentRun=null,busy=null,runDocument='';
      const token=location.hash.slice(1)||sessionStorage.getItem('optiprop-token')||'';
      if(location.hash){sessionStorage.setItem('optiprop-token',token);historyReplace();}
      function historyReplace(){window.history.replaceState(null,'',location.pathname);}
      async function api(path,data){const response=await fetch(path,{method:data===undefined?'GET':'POST',headers:{'X-OptiProp-Token':token,'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data)});const result=await response.json();if(!response.ok)throw Error(result.error||'Request failed');return result;}
      function guarded(fn){return async()=>{try{await fn();}catch(e){message(e.message);}};}
      const documentKey=()=>JSON.stringify(project);
      const canObserve=()=>!!currentRun&&!busy&&runDocument===documentKey();
      function controls(){q('#gb-run').disabled=!!busy;q('#gb-cancel').disabled=!busy;q('#gb-check').disabled=!!busy;['#gb-view','#gb-export','#gb-xz'].forEach(id=>q(id).disabled=!canObserve());if(currentRun&&!busy&&runDocument!==documentKey())q('#gb-run-status').textContent='草稿已改動；下圖仍是先前的計算結果。請重新計算。';else if(canObserve()&&q('#gb-run-status').textContent.startsWith('草稿已改動'))q('#gb-run-status').textContent='目前設定符合已完成的 Run '+currentRun.slice(0,8)+'。';}
      const layers=()=>project.optical_system.layers;
      const index=()=>layers().findIndex(x=>x.id===selected);
      const selectedNode=()=>index()<0?project.source:layers()[index()];
      const real=value=>typeof value==='object'&&value!==null?Number(value.real||0):Number(value??1);
      const um=n=>Number((n*1e6).toFixed(6));
      const message=text=>{q('#gb-message').hidden=!text;q('#gb-message').textContent=text;};
      function snapshot(){history.push({project:clone(project),selected});if(history.length>30)history.shift();}
      const sourceZ=()=>Number(project.source.imported_field?.z_m||0);
      function positions(){let z=sourceZ(),n=real(project.source.medium_index);return layers().map(layer=>{const row={before:z,nBefore:n};if(layer.enabled!==false){if(layer.type==='PropagationLayer'){z+=layer.spec.distance_m;n=real(layer.spec.medium_index??n);}if(layer.type==='InterfaceLayer')n=real(layer.n2);}return{...row,after:z,nAfter:n};});}
      function change(fn){snapshot();fn();message('');render();controls();}
      function switchLens(node,type){
        project.ui_state??={};project.ui_state.lens_model_drafts??={};
        const drafts=project.ui_state.lens_model_drafts[node.id]??={};
        drafts[node.type]=Object.fromEntries(modelFields[node.type].filter(key=>node[key]!==undefined).map(key=>[key,clone(node[key])]));
        const saved=drafts[type]||modelDefaults[type];
        Object.values(modelFields).flat().forEach(key=>delete node[key]);
        node.type=type;
        modelFields[type].filter(key=>saved[key]!==undefined).forEach(key=>node[key]=clone(saved[key]));
      }
      function field(label,key,value,apply,options={}){
        const wrap=document.createElement('label');wrap.className='gb-field';wrap.textContent=label;
        const input=document.createElement('input');input.id='gb-edit-'+key;input.type=options.text?'text':'number';input.value=value;input.required=true;
        if(!options.text)input.step=options.integer?'1':'any';if(options.min!==undefined)input.min=options.min;if(options.max!==undefined)input.max=options.max;
        input.addEventListener('change',()=>{if(!input.reportValidity())return;const next=options.text?input.value:Number(input.value);if(options.nonzero&&next===0){input.setCustomValidity('不可為零');input.reportValidity();input.setCustomValidity('');return;}change(()=>apply(next));});
        wrap.appendChild(input);q('#gb-editor').appendChild(wrap);
      }
      function renderEditor(){
        const node=selectedNode(),i=index(),editor=q('#gb-editor');editor.replaceChildren();q('#gb-editor-title').textContent=kinds[node.type]||'光源';
        field('名稱','name',node.name,v=>node.name=v,{text:true});
        if(i<0){
          if(node.kind!=='imported_field'){
            const label=document.createElement('label');label.className='gb-field';label.textContent='光源類型';const select=document.createElement('select');select.id='gb-source-kind';
            ['gaussian','plane_wave','tilted_plane_wave'].forEach(kind=>{const option=document.createElement('option');option.value=kind;option.textContent=kind;select.appendChild(option);});select.value=node.kind;select.addEventListener('change',()=>change(()=>{node.kind=select.value;if(node.kind==='gaussian'){node.waist_x_m??=10e-6;node.waist_y_m??=10e-6;}}));label.appendChild(select);editor.appendChild(label);
            field('真空波長 · nm','wavelength',node.wavelength_m*1e9,v=>node.wavelength_m=v*1e-9,{min:.001});field('來源折射率','source-n',real(node.medium_index),v=>node.medium_index=v,{min:.001});
            ['nx','ny'].forEach(key=>field(key.toUpperCase(),key,node.grid[key],v=>node.grid[key]=v,{min:2,max:4096,integer:true}));
            ['dx','dy'].forEach(key=>field(key+' · µm',key,um(node.grid[key]),v=>node.grid[key]=v*1e-6,{min:.001}));
            node.components.forEach((name,j)=>field(name+' 入射振幅（實數）','amp'+j,real(node.amplitudes[j]),v=>node.amplitudes[j]=v));
            if(node.kind==='gaussian'){['x','y'].forEach(axis=>field('束腰 w'+axis+' · µm','waist-'+axis,um(node['waist_'+axis+'_m']),v=>node['waist_'+axis+'_m']=v*1e-6,{min:.001}));}
            if(node.kind==='tilted_plane_wave'){['x','y'].forEach(axis=>field('傾角 '+axis+' · rad','angle-'+axis,node['angle_'+axis+'_rad']||0,v=>node['angle_'+axis+'_rad']=v));}
          }
          const note=document.createElement('div');note.className='gb-muted';note.textContent=node.kind+' · '+(node.components||['scalar']).join(' / ')+' · '+(node.grid?node.grid.nx+' × '+node.grid.ny:'外部光場')+(node.kind==='imported_field'?'（取樣與波長依原始檔）':'');editor.appendChild(note);
        }
        if(node.type==='PropagationLayer'){
          field('傳遞距離 · µm','distance',um(node.spec.distance_m),v=>node.spec.distance_m=v*1e-6,{min:0});field('這一段的折射率','index',real(node.spec.medium_index),v=>node.spec.medium_index=v,{min:.001});
          const label=document.createElement('label');label.className='gb-field';label.textContent='傳遞方法';const select=document.createElement('select');select.id='gb-method';['asm','blas','fresnel_tf','rs_fft'].forEach(method=>{const option=document.createElement('option');option.value=method;option.textContent=method.toUpperCase();select.appendChild(option);});if(![...select.options].some(o=>o.value===node.spec.method)){const option=document.createElement('option');option.value=node.spec.method;option.textContent=node.spec.method+'（唯讀參考）';select.appendChild(option);}select.value=node.spec.method;select.addEventListener('change',()=>change(()=>node.spec.method=select.value));label.appendChild(select);editor.appendChild(label);
        }
        if(modelFields[node.type]){
          const label=document.createElement('label');label.className='gb-field';label.textContent='透鏡相位類型';
          const select=document.createElement('select');select.id='gb-lens-model';
          [['IdealLensLayer',node.phase_model==='paraxial'?'理想相位 · 近軸':'理想相位 · 等光程'],['Binary2LensLayer','Binary2 · 徑向多項式']].forEach(([value,text])=>{const option=document.createElement('option');option.value=value;option.textContent=text;select.appendChild(option);});
          select.value=node.type;select.addEventListener('change',()=>change(()=>switchLens(node,select.value)));label.appendChild(select);editor.appendChild(label);
          if(node.type==='IdealLensLayer'){
            field('參考焦長 · µm','focal',um(node.focal_length_m),v=>node.focal_length_m=v*1e-6,{nonzero:true});
            if(node.design_wavelength_m)field('相位設計波長 · nm','design-wavelength',node.design_wavelength_m*1e9,v=>node.design_wavelength_m=v*1e-9,{min:.001});
          }else{
            node.coefficients.forEach((value,j)=>field('C'+(j+1)+' · r^'+(2*j+2)+' 項 · rad/mm^'+(2*j+2),'c'+(j+1),value,v=>node.coefficients[j]=v));
            const controls=document.createElement('div');controls.className='gb-controls';
            const add=document.createElement('button');add.type='button';add.id='gb-add-coefficient';add.textContent='＋ 增加高階項';add.disabled=node.coefficients.length>=32;add.addEventListener('click',()=>change(()=>node.coefficients.push(0)));
            const remove=document.createElement('button');remove.type='button';remove.id='gb-remove-coefficient';remove.textContent='移除最高階';remove.disabled=node.coefficients.length===1;remove.addEventListener('click',()=>change(()=>node.coefficients.pop()));
            controls.append(add,remove);editor.appendChild(controls);
            const note=document.createElement('div');note.className='gb-muted';note.textContent=node.coefficients.every(c=>c===0)?'全 0：平坦相位，尚未設定聚焦係數。':'固定弧度相位 · 半徑單位 mm · 不自動換算波長';editor.appendChild(note);
          }
          const pupilLabel=document.createElement('label');pupilLabel.className='gb-check';const pupil=document.createElement('input');pupil.id='gb-pupil';pupil.type='checkbox';pupil.checked=!!node.aperture;pupil.addEventListener('change',()=>change(()=>node.aperture=pupil.checked?{shape:'circle',size_x_m:60e-6}:null));pupilLabel.append(pupil,'限制透鏡孔徑');editor.appendChild(pupilLabel);
          if(node.aperture?.shape==='circle')field('透鏡孔徑直徑 · µm','diameter',um(node.aperture.size_x_m),v=>node.aperture.size_x_m=v*1e-6,{min:.001});
          const note=document.createElement('div');note.className='gb-muted';note.textContent='Ex / Ey 共用相位。切換保留兩組設定，不改下游傳遞距離。';editor.appendChild(note);
        }
        if(node.type==='ApertureLayer')field('孔徑尺寸 x · µm','aperture',um(node.aperture.size_x_m),v=>node.aperture.size_x_m=v*1e-6,{min:.001});
        if(node.type==='InterfaceLayer'){
          field('入射介質 n₁','n1',real(node.n1),v=>node.n1=v,{min:.001});field('出射介質 n₂','n2',real(node.n2),v=>node.n2=v,{min:.001});
          const label=document.createElement('label');label.className='gb-check';const check=document.createElement('input');check.type='checkbox';check.checked=!!node.ignore_reflection;check.addEventListener('change',()=>change(()=>node.ignore_reflection=check.checked));label.append(check,'忽略反射（明確假設）');editor.appendChild(label);
        }
        if(i>=0){const label=document.createElement('label');label.className='gb-check';const check=document.createElement('input');check.id='gb-enabled';check.type='checkbox';check.checked=node.enabled!==false;check.addEventListener('change',()=>change(()=>node.enabled=check.checked));label.append(check,node.type==='PropagationLayer'?'啟用此段（停用時也略過距離）':'啟用此元件');editor.appendChild(label);}
        q('#gb-layer-controls').hidden=i<0;q('#gb-up').disabled=i<=0;q('#gb-down').disabled=i<0||i>=layers().length-1;
      }
      function render(){
        root.style.setProperty('--gb-row',appearance.spacing+'px');
        const rows=positions(),list=q('#gb-stack');list.replaceChildren();
        [project.source,...layers()].forEach((node,j)=>{const button=document.createElement('button');button.type='button';button.className='gb-node';button.dataset.id=node.id;button.setAttribute('aria-pressed',String(node.id===selected));
          const order=document.createElement('span');order.className='gb-index';order.textContent=j===0?'S':String(j).padStart(2,'0');const name=document.createElement('span');name.className='gb-name';name.textContent=node.name;
          const meta=document.createElement('span');meta.className='gb-meta';const row=rows[j-1];meta.textContent=j===0?'光源 · z = '+um(sourceZ())+' µm':(node.enabled===false?'略過 · ':'')+kinds[node.type]+' · z '+um(row.before)+' → '+um(row.after)+' µm · n '+row.nBefore+' → '+row.nAfter;
          button.append(order,name,meta);button.addEventListener('click',()=>{selected=node.id;render();});list.appendChild(button);
        });
        q('#gb-empty').hidden=layers().length>0;q('#gb-length').textContent=layers().length+' 個步驟 · 終點 z = '+um(rows.at(-1)?.after??sourceZ())+' µm';q('#gb-project-name').textContent=project.name;
        q('#gb-undo').disabled=history.length===0;q('#gb-json').textContent=JSON.stringify(project,null,2);q('#gb-device').value=project.compute_settings?.device||'cpu';q('#gb-precision').value=project.compute_settings?.precision||'complex128';renderEditor();renderObserver();controls();
      }
      function renderObserver(){const i=index(),row=positions()[i],before=q('#gb-side').value==='before';const z=i<0?sourceZ():before?row.before:row.after;const n=i<0?real(project.source.medium_index):before?row.nBefore:row.nAfter;q('#gb-observer').textContent=selectedNode().name+' · '+(i<0?'來源':before?'輸入':'輸出')+' · z = '+um(z)+' µm · n = '+n;}
      function insert(type){const i=index(),rows=positions(),n=i<0?real(project.source.medium_index):rows[i].nAfter;change(()=>{const node=type==='PropagationLayer'?prop(100e-6,n):element(type,n);layers().splice(i+1,0,node);selected=node.id;});}
      function load(doc){if(doc.format!=='optiprop-project'||doc.schema_version!==1||doc.optical_system?.type!=='OpticalSystem'||!Array.isArray(doc.optical_system.layers)||!doc.source)throw Error('需要 canonical optiprop-project v1 JSON。');
        const nodes=[doc.source,...doc.optical_system.layers],ids=nodes.map(x=>x.id);if(ids.some(id=>typeof id!=='string')||new Set(ids).size!==ids.length)throw Error('各步驟需要不重複的 ID。');
        if(doc.optical_system.layers.some(x=>!kinds[x.type]))throw Error('工作台尚未提供這種元件的編輯器；不會丟棄或改寫檔案。');
        if(doc.optical_system.layers.some(x=>x.type==='PropagationLayer'&&(!x.spec||!Number.isFinite(x.spec.distance_m))))throw Error('傳遞段缺少合法 distance_m。');
        if(doc.optical_system.layers.some(x=>x.type==='Binary2LensLayer'&&(!Array.isArray(x.coefficients)||!x.coefficients.length||x.coefficients.some(c=>!Number.isFinite(c)))))throw Error('Binary2 需要非空、有限數值的 coefficients。');
        change(()=>{project=clone(doc);selected=project.source.id;});
        if(project.assets?.length)message('光路已載入；請在「本機檔案與計算設定」指定資產根目錄，再檢查或計算。');
      }
      q('#gb-add-propagation').addEventListener('click',()=>insert('PropagationLayer'));q('#gb-add-element').addEventListener('click',()=>insert(q('#gb-add-kind').value));
      q('#gb-up').addEventListener('click',()=>{const i=index();if(i>0)change(()=>[layers()[i-1],layers()[i]]=[layers()[i],layers()[i-1]]);});
      q('#gb-down').addEventListener('click',()=>{const i=index();if(i>=0&&i<layers().length-1)change(()=>[layers()[i],layers()[i+1]]=[layers()[i+1],layers()[i]]);});
      q('#gb-copy').addEventListener('click',()=>{if(index()<0)return;change(()=>{const i=index(),copy=clone(selectedNode()),drafts=project.ui_state?.lens_model_drafts;if(drafts?.[copy.id])drafts[copy.id=uid()]=clone(drafts[selected]);else copy.id=uid();copy.name+=' · 副本';layers().splice(i+1,0,copy);selected=copy.id;});});
      q('#gb-remove').addEventListener('click',()=>{const i=index();if(i>=0)change(()=>{if(project.ui_state?.lens_model_drafts)delete project.ui_state.lens_model_drafts[selected];layers().splice(i,1);selected=layers()[Math.max(0,i-1)]?.id||project.source.id;});});
      q('#gb-undo').addEventListener('click',()=>{const last=history.pop();if(last){project=last.project;selected=last.selected;message('已復原');render();}});
      q('#gb-side').addEventListener('change',renderObserver);
      q('#gb-new').addEventListener('click',()=>{change(()=>{project=blank();selected=project.source.id;});q('#gb-example').value='';});
      q('#gb-example').addEventListener('change',()=>{
        const mapping={free:'free_propagation',focus:'metalens_focus_ideal','focus-binary2':'metalens_focus_binary2',laser:'laser_collimator_demo'};
        const key=mapping[q('#gb-example').value];if(!key)return;
        if(!examples[key]){message('範例尚未載入，請重新啟動工作台。');return;}
        load(examples[key]);
        if(key==='laser_collimator_demo')message('解析光源示範，不是實際 P3_10um 光場或已驗證的準直設計。');
      });
      q('#gb-open').addEventListener('click',()=>q('#gb-file').click());q('#gb-file').addEventListener('change',async()=>{try{const file=q('#gb-file').files[0];if(!file)return;if(file.size>1024*1024)throw Error('此草稿只載入 1 MB 以下的設定 JSON，不載入大型場資料。');load(JSON.parse(await file.text()));q('#gb-example').value='';}catch(e){message('未載入：'+e.message);}finally{q('#gb-file').value='';}});
      const runPayload=()=>({project:clone(project),asset_root:q('#gb-asset-root').value});
      const observation=()=>{if(!canObserve())throw Error('請先計算目前光路。');return {job:currentRun,layer_id:selected,side:q('#gb-side').value,quantity:q('#gb-quantity').value};};
      async function showView(){
        const selection=observation(),result=await api('/api/view',selection);
        if(selection.job!==currentRun||!canObserve())return;
        q('#gb-result').src='data:image/png;base64,'+result.png;q('#gb-result').hidden=false;
        const viewedNode=selection.layer_id===project.source.id?project.source:layers().find(x=>x.id===selection.layer_id);
        q('#gb-result-plane').textContent='Run '+currentRun.slice(0,8)+' · '+(viewedNode?.name||selection.layer_id)+' · '+selection.side+' · z = '+um(result.field.z_m)+' µm · '+result.field.nx+' × '+result.field.ny+' · '+result.field.dtype;
      }
      async function waitJob(id){
        while(true){const job=await api('/api/jobs/'+id);q('#gb-run-status').textContent=job.message+(job.output_dir?' · '+job.output_dir:'');if(!['queued','running'].includes(job.status)){if(job.status!=='completed')throw Error(job.message);return job;}await new Promise(resolve=>setTimeout(resolve,500));}
      }
      q('#gb-check').addEventListener('click',guarded(async()=>{q('#gb-check').disabled=true;try{const result=await api('/api/validate',runPayload());message((result.valid?'檢查通過':'設定有錯誤')+(result.issues.length?'：'+result.issues.map(i=>i.severity+' — '+i.message).join('；'):'。仍需依問題做取樣收斂驗證。'));}finally{controls();}}));
      q('#gb-run').addEventListener('click',guarded(async()=>{
        const payload=runPayload(),key=documentKey();
        try{busy='starting';controls();const start=await api('/api/run',payload);busy=start.id;controls();const job=await waitJob(busy);currentRun=job.id;runDocument=key;busy=null;controls();if(canObserve())await showView();if(job.issues?.length)message(job.issues.map(i=>i.message).join('；'));}finally{busy=null;controls();}
      }));
      q('#gb-cancel').addEventListener('click',guarded(async()=>{if(busy&&busy!=='starting')await api('/api/cancel',{job:busy});}));
      q('#gb-view').addEventListener('click',guarded(showView));
      q('#gb-export').addEventListener('click',guarded(async()=>{const result=await api('/api/export',{...observation(),format:q('#gb-format').value});message('已儲存：'+result.path);}));
      q('#gb-xz').addEventListener('click',guarded(async()=>{
        if(!q('#gb-xz-distance').reportValidity()||!q('#gb-xz-count').reportValidity())return;
        const payload={...observation(),distance_m:Number(q('#gb-xz-distance').value)*1e-6,count:Number(q('#gb-xz-count').value)};
        try{busy='starting';controls();const start=await api('/api/xz',payload);busy=start.id;controls();await waitJob(busy);const result=await api('/api/xz-image',{job:busy});q('#gb-xz-result').src='data:image/png;base64,'+result.png;q('#gb-xz-result').hidden=false;}finally{busy=null;controls();}
      }));
      q('#gb-save').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(project,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='optiprop-project.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
      q('#gb-device').addEventListener('change',()=>change(()=>project.compute_settings.device=q('#gb-device').value));
      q('#gb-precision').addEventListener('change',()=>change(()=>project.compute_settings.precision=q('#gb-precision').value));
      q('#gb-import').addEventListener('click',guarded(async()=>{if(!q('#gb-import-path').value.trim())throw Error('請輸入光場完整路徑。');const result=await api('/api/import-source',{path:q('#gb-import-path').value.trim()});q('#gb-asset-root').value=result.asset_root;load(result.project);message('已建立 canonical 資產副本：'+result.asset_root);}));
      q('#gb-quit').addEventListener('click',guarded(async()=>{if(!confirm('結束本機工作台？請先儲存尚未保存的 JSON；執行中的工作會取消。'))return;await api('/api/shutdown',{});root.replaceChildren();const p=document.createElement('p');p.textContent='OptiProp 已結束，可以關閉此分頁。';root.appendChild(p);}));
      window.addEventListener('beforeunload',event=>{if(busy){event.preventDefault();event.returnValue='';}});
      render();
      api('/api/bootstrap').then(result=>{examples=result.examples;q('#gb-version').textContent='v'+result.version+' · 本機運算 · 無 Qt';q('#gb-asset-root').value=result.asset_root;q('#gb-output-root').textContent='結果位置：'+result.output_root;if(result.initial_project)load(result.initial_project);}).catch(e=>{message('無法連線本機工作台：'+e.message);q('#gb-run').disabled=true;});
    })();
