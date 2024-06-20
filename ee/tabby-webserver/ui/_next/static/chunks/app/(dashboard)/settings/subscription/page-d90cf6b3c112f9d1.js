(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4303],{47714:function(e,t,n){Promise.resolve().then(n.bind(n,47166))},47166:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return E}});var r=n(36164),s=n(88542),i=n(99092),a=n.n(i),l=n(18500),o=n(29917),c=n(3448),u=n(6230),d=n(73051),f=n(3546),m=n(84381),x=n(5493),p=n(2578),h=n(23782),j=n(43240),v=n(24449),b=n(11634),g=n(74248),y=n(73460),N=n(31458),w=n(98150),Z=n(81565);let R=f.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("textarea",{className:(0,g.cn)("flex min-h-[80px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",n),ref:t,...s})});R.displayName="Textarea";let S=h.Ry({license:h.Z_()}),C=(0,j.BX)("\n  mutation UploadLicense($license: String!) {\n    uploadLicense(license: $license)\n  }\n"),O=(0,j.BX)("\n  mutation ResetLicense {\n    resetLicense\n  }\n");function k(e){let{className:t,onSuccess:n,canReset:s,...i}=e,a=(0,x.cI)({resolver:(0,m.F)(S)}),l=a.watch("license"),[o,c]=f.useState(!1),[u,d]=f.useState(!1),[h,j]=f.useState(!1),k=(0,v.S)((e,t)=>{c(e),t&&(a.reset({license:""}),p.A.success("License is uploaded"),null==n||n())},500,{leading:!0}),T=(0,v.S)((e,t)=>{j(e),t&&(d(!1),null==n||n())},500,{leading:!0}),A=(0,b.D)(C,{form:a}),L=(0,b.D)(O);return(0,r.jsx)("div",{className:(0,g.cn)(t),...i,children:(0,r.jsx)(w.l0,{...a,children:(0,r.jsxs)("form",{className:"grid gap-6",onSubmit:a.handleSubmit(e=>(k.run(!0),A(e).then(e=>{var t;k.run(!1,null==e?void 0:null===(t=e.data)||void 0===t?void 0:t.uploadLicense)}))),children:[(0,r.jsx)(w.Wi,{control:a.control,name:"license",render:e=>{let{field:t}=e;return(0,r.jsxs)(w.xJ,{children:[(0,r.jsx)(w.NI,{children:(0,r.jsx)(R,{className:"min-h-[200px]",placeholder:"Paste your license here - write only",...t})}),(0,r.jsx)(w.zG,{})]})}}),(0,r.jsxs)("div",{className:"flex items-start justify-between gap-4",children:[(0,r.jsx)("div",{children:(0,r.jsx)(w.zG,{})}),(0,r.jsxs)("div",{className:"flex shrink-0 items-center gap-4",children:[(0,r.jsxs)(y.aR,{open:u,onOpenChange:e=>{h||d(e)},children:[s&&(0,r.jsx)(y.vW,{asChild:!0,children:(0,r.jsx)(N.z,{type:"button",variant:"hover-destructive",children:"Reset"})}),(0,r.jsxs)(y._T,{children:[(0,r.jsxs)(y.fY,{children:[(0,r.jsx)(y.f$,{children:"Are you absolutely sure?"}),(0,r.jsx)(y.yT,{children:"This action cannot be undone. It will reset the current license."})]}),(0,r.jsxs)(y.xo,{children:[(0,r.jsx)(y.le,{children:"Cancel"}),(0,r.jsxs)(y.OL,{className:(0,N.d)({variant:"destructive"}),onClick:e=>{e.preventDefault(),T.run(!0),L().then(e=>{var t,n;let r=null==e?void 0:null===(t=e.data)||void 0===t?void 0:t.resetLicense;T.run(!1,r),(null==e?void 0:e.error)&&p.A.error(null!==(n=e.error.message)&&void 0!==n?n:"reset failed")})},disabled:h,children:[h&&(0,r.jsx)(Z.IconSpinner,{className:"mr-2 h-4 w-4 animate-spin"}),"Yes, reset it"]})]})]})]}),(0,r.jsxs)(N.z,{type:"submit",disabled:o||!l,children:[o&&(0,r.jsx)(Z.IconSpinner,{className:"mr-2 h-4 w-4 animate-spin"}),"Upload License"]})]})]})]})})})}var T=n(99047),A=n(29);let L=()=>(0,r.jsxs)(T.iA,{className:"border text-center",children:[(0,r.jsx)(T.xD,{children:(0,r.jsxs)(T.SC,{children:[(0,r.jsx)(T.ss,{className:"w-[40%]"}),D.map((e,t)=>{let{name:n,pricing:s,limit:i}=e;return(0,r.jsxs)(T.ss,{className:"w-[20%] text-center",children:[(0,r.jsx)("h1",{className:"py-4 text-2xl font-bold",children:n}),(0,r.jsx)("p",{className:"text-center font-semibold",children:s}),(0,r.jsx)("p",{className:"pb-2 pt-1",children:i})]},t)})]})}),(0,r.jsx)(T.RM,{children:U.map((e,t)=>{let{name:n,features:s}=e;return(0,r.jsx)(z,{name:n,features:s},t)})})]}),z=e=>{let{name:t,features:n}=e;return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(T.SC,{children:(0,r.jsx)(T.pj,{colSpan:4,className:"bg-accent text-left text-accent-foreground",children:t})}),n.map((e,t)=>{let{name:n,community:s,team:i,enterprise:a}=e;return(0,r.jsxs)(T.SC,{children:[(0,r.jsx)(T.pj,{className:"text-left",children:n}),(0,r.jsx)(T.pj,{className:"font-semibold",children:s}),(0,r.jsx)(T.pj,{className:"font-semibold",children:i}),(0,r.jsx)(T.pj,{className:"font-semibold text-primary",children:a})]},t)})]})},D=[{name:"Community",pricing:"$0 per user/month",limit:"Up to 5 users, single node"},{name:"Team",pricing:"$19 per user/month",limit:"Up to 30 users, up to 2 nodes"},{name:"Enterprise",pricing:"Contact Us",limit:"Customized, billed annually"}],I=e=>{let{children:t}=e;return(0,r.jsx)(A.pn,{children:(0,r.jsxs)(A.u,{children:[(0,r.jsx)(A.aJ,{children:(0,r.jsx)(Z.IconInfoCircled,{})}),(0,r.jsx)(A._v,{children:(0,r.jsx)("p",{className:"max-w-[320px]",children:t})})]})})},_=e=>{let{name:t,children:n}=e;return(0,r.jsxs)("span",{className:"flex gap-1",children:[t,(0,r.jsx)(I,{children:n})]})},F=(0,r.jsx)(Z.IconCheck,{className:"mx-auto"}),U=[{name:"Features",features:[{name:"User count",community:"Up to 5",team:"Up to 30",enterprise:"Unlimited"},{name:"Secure Access",community:F,team:F,enterprise:F},{name:(0,r.jsx)(_,{name:"Bring your own LLM",children:"Tabby builds on top of open technologies, allowing customers to bring their own LLM models."}),community:F,team:F,enterprise:F},{name:(0,r.jsx)(_,{name:"Git Providers",children:"Tabby can retrieve the codebase context to enhance responses. Underlying Tabby pulls context from git providers with a code search index. This method enables Tabby to utilize the team's past practices at scale."}),community:F,team:F,enterprise:F},{name:"Usage Reports and Analytics",community:F,team:F,enterprise:F},{name:"Admin Controls",community:"–",team:F,enterprise:F},{name:"Toggle IDE / Extensions telemetry",community:"–",team:"–",enterprise:F},{name:"Authentication Domain",community:"–",team:"–",enterprise:F},{name:"Single Sign-On (SSO)",community:"–",team:"–",enterprise:F}]},{name:"Bespoke",features:[{name:"Support",community:"Community",team:"Email",enterprise:"Dedicated Slack channel"},{name:"Roadmap prioritization",community:"–",team:"–",enterprise:F}]}];function E(){let[{data:e,fetching:t},n]=(0,o.jp)(),s=null==e?void 0:e.license,i=!!(null==s?void 0:s.type)&&s.type!==l.oj.Community;return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(d.b,{className:"mb-8",externalLink:"https://links.tabbyml.com/schedule-a-demo",externalLinkText:"\uD83D\uDCC6 Book a 30-minute product demo",children:"You can upload your Tabby license to unlock team/enterprise features."}),(0,r.jsxs)("div",{className:"flex flex-col gap-8",children:[(0,r.jsx)(u.Z,{loading:t,fallback:(0,r.jsxs)("div",{className:"grid grid-cols-3",children:[(0,r.jsx)(c.O,{className:"h-16 w-[80%]"}),(0,r.jsx)(c.O,{className:"h-16 w-[80%]"}),(0,r.jsx)(c.O,{className:"h-16 w-[80%]"})]}),children:s&&(0,r.jsx)(Y,{license:s})}),(0,r.jsx)(k,{onSuccess:()=>{n()},canReset:i}),(0,r.jsx)(L,{})]})]})}function Y(e){var t;let{license:n}=e,i=n.expiresAt?a()(n.expiresAt).format("MM/DD/YYYY"):"–",l="".concat(n.seatsUsed," / ").concat(n.seats);return(0,r.jsxs)("div",{className:"grid font-bold lg:grid-cols-3",children:[(0,r.jsxs)("div",{children:[(0,r.jsx)("div",{className:"mb-1 text-muted-foreground",children:"Expires at"}),(0,r.jsx)("div",{className:"text-3xl",children:i})]}),(0,r.jsxs)("div",{children:[(0,r.jsx)("div",{className:"mb-1 text-muted-foreground",children:"Assigned / Total Seats"}),(0,r.jsx)("div",{className:"text-3xl",children:l})]}),(0,r.jsxs)("div",{children:[(0,r.jsx)("div",{className:"mb-1 text-muted-foreground",children:"Current plan"}),(0,r.jsx)("div",{className:"text-3xl text-primary",children:(0,s.Z)(null!==(t=null==n?void 0:n.type)&&void 0!==t?t:"Community")})]})]})}},6230:function(e,t,n){"use strict";var r=n(36164),s=n(3546),i=n(24449),a=n(90379);t.Z=e=>{let{loading:t,fallback:n,delay:l,children:o}=e,[c,u]=s.useState(!t),[d]=(0,i.n)(c,null!=l?l:200);return(s.useEffect(()=>{t||c||u(!0)},[t]),d)?o:n||(0,r.jsx)(a.cg,{})}},90379:function(e,t,n){"use strict";n.d(t,{PF:function(){return o},cg:function(){return a},tB:function(){return l}});var r=n(36164),s=n(74248),i=n(3448);let a=e=>{let{className:t,...n}=e;return(0,r.jsxs)("div",{className:(0,s.cn)("space-y-3",t),...n,children:[(0,r.jsx)(i.O,{className:"h-4 w-full"}),(0,r.jsx)(i.O,{className:"h-4 w-full"}),(0,r.jsx)(i.O,{className:"h-4 w-full"}),(0,r.jsx)(i.O,{className:"h-4 w-full"})]})},l=e=>{let{className:t,...n}=e;return(0,r.jsx)(i.O,{className:(0,s.cn)("h-4 w-full",t),...n})},o=e=>{let{className:t,...n}=e;return(0,r.jsxs)("div",{className:(0,s.cn)("flex flex-col gap-3",t),...n,children:[(0,r.jsx)(i.O,{className:"h-4 w-[20%]"}),(0,r.jsx)(i.O,{className:"h-4 w-full"}),(0,r.jsx)(i.O,{className:"h-4 w-[20%]"}),(0,r.jsx)(i.O,{className:"h-4 w-full"})]})}},73051:function(e,t,n){"use strict";n.d(t,{b:function(){return o}});var r=n(36164);n(3546);var s=n(70652),i=n.n(s),a=n(74248),l=n(81565);let o=e=>{let{className:t,externalLink:n,externalLinkText:s="Learn more",children:o}=e;return(0,r.jsx)("div",{className:(0,a.cn)("mb-4 flex items-center gap-4",t),children:(0,r.jsxs)("div",{className:"flex-1 text-sm text-muted-foreground",children:[o,!!n&&(0,r.jsxs)(i(),{className:"ml-2 inline-flex cursor-pointer flex-row items-center text-primary hover:underline",href:n,target:"_blank",children:[s,(0,r.jsx)(l.IconExternalLink,{className:"ml-1"})]})]})})}},73460:function(e,t,n){"use strict";n.d(t,{OL:function(){return j},_T:function(){return f},aR:function(){return o},f$:function(){return p},fY:function(){return m},le:function(){return v},vW:function(){return c},xo:function(){return x},yT:function(){return h}});var r=n(36164),s=n(3546),i=n(28961),a=n(74248),l=n(31458);let o=i.fC,c=i.xz,u=e=>{let{className:t,children:n,...s}=e;return(0,r.jsx)(i.h_,{className:(0,a.cn)(t),...s,children:(0,r.jsx)("div",{className:"fixed inset-0 z-50 flex items-end justify-center sm:items-center",children:n})})};u.displayName=i.h_.displayName;let d=s.forwardRef((e,t)=>{let{className:n,children:s,...l}=e;return(0,r.jsx)(i.aV,{className:(0,a.cn)("fixed inset-0 z-50 bg-background/80 backdrop-blur-sm transition-opacity animate-in fade-in",n),...l,ref:t})});d.displayName=i.aV.displayName;let f=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsxs)(u,{children:[(0,r.jsx)(d,{}),(0,r.jsx)(i.VY,{ref:t,className:(0,a.cn)("fixed z-50 grid w-full max-w-lg scale-100 gap-4 border bg-background p-6 opacity-100 shadow-lg animate-in fade-in-90 slide-in-from-bottom-10 sm:rounded-lg sm:zoom-in-90 sm:slide-in-from-bottom-0 md:w-full",n),...s})]})});f.displayName=i.VY.displayName;let m=e=>{let{className:t,...n}=e;return(0,r.jsx)("div",{className:(0,a.cn)("flex flex-col space-y-2 text-center sm:text-left",t),...n})};m.displayName="AlertDialogHeader";let x=e=>{let{className:t,...n}=e;return(0,r.jsx)("div",{className:(0,a.cn)("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2",t),...n})};x.displayName="AlertDialogFooter";let p=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)(i.Dx,{ref:t,className:(0,a.cn)("text-lg font-semibold",n),...s})});p.displayName=i.Dx.displayName;let h=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)(i.dk,{ref:t,className:(0,a.cn)("text-sm text-muted-foreground",n),...s})});h.displayName=i.dk.displayName;let j=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)(i.aU,{ref:t,className:(0,a.cn)((0,l.d)(),n),...s})});j.displayName=i.aU.displayName;let v=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)(i.$j,{ref:t,className:(0,a.cn)((0,l.d)({variant:"outline"}),"mt-2 sm:mt-0",n),...s})});v.displayName=i.$j.displayName},31458:function(e,t,n){"use strict";n.d(t,{d:function(){return o},z:function(){return c}});var r=n(36164),s=n(3546),i=n(74047),a=n(14375),l=n(74248);let o=(0,a.j)("inline-flex items-center justify-center rounded-md text-sm font-medium shadow ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",{variants:{variant:{default:"bg-primary text-primary-foreground shadow-md hover:bg-primary/90",destructive:"bg-destructive text-destructive-foreground hover:bg-destructive/90","hover-destructive":"shadow-none hover:bg-destructive/90 hover:text-destructive-foreground",outline:"border border-input hover:bg-accent hover:text-accent-foreground",secondary:"bg-secondary text-secondary-foreground hover:bg-secondary/80",ghost:"shadow-none hover:bg-accent hover:text-accent-foreground",link:"text-primary underline-offset-4 shadow-none hover:underline"},size:{default:"h-8 px-4 py-2",sm:"h-8 rounded-md px-3",lg:"h-11 rounded-md px-8",icon:"h-8 w-8 p-0"}},defaultVariants:{variant:"default",size:"default"}}),c=s.forwardRef((e,t)=>{let{className:n,variant:s,size:a,asChild:c=!1,...u}=e,d=c?i.g7:"button";return(0,r.jsx)(d,{className:(0,l.cn)(o({variant:s,size:a,className:n})),ref:t,...u})});c.displayName="Button"},98150:function(e,t,n){"use strict";n.d(t,{NI:function(){return h},Wi:function(){return d},l0:function(){return c},lX:function(){return p},pf:function(){return j},xJ:function(){return x},zG:function(){return v}});var r=n(36164),s=n(3546),i=n(74047),a=n(5493),l=n(74248),o=n(5266);let c=a.RV,u=s.createContext({}),d=e=>{let{...t}=e;return(0,r.jsx)(u.Provider,{value:{name:t.name},children:(0,r.jsx)(a.Qr,{...t})})},f=()=>{let e=s.useContext(u),t=s.useContext(m),{getFieldState:n,formState:r}=(0,a.Gc)(),i=e.name||"root",l=n(i,r);if(!r)throw Error("useFormField should be used within <Form>");let{id:o}=t;return{id:o,name:i,formItemId:"".concat(o,"-form-item"),formDescriptionId:"".concat(o,"-form-item-description"),formMessageId:"".concat(o,"-form-item-message"),...l}},m=s.createContext({}),x=s.forwardRef((e,t)=>{let{className:n,...i}=e,a=s.useId();return(0,r.jsx)(m.Provider,{value:{id:a},children:(0,r.jsx)("div",{ref:t,className:(0,l.cn)("space-y-2",n),...i})})});x.displayName="FormItem";let p=s.forwardRef((e,t)=>{let{className:n,required:s,...i}=e,{error:a,formItemId:c}=f();return(0,r.jsx)(o._,{ref:t,className:(0,l.cn)(a&&"text-destructive",s&&'after:ml-0.5 after:text-destructive after:content-["*"]',n),htmlFor:c,...i})});p.displayName="FormLabel";let h=s.forwardRef((e,t)=>{let{...n}=e,{error:s,formItemId:a,formDescriptionId:l,formMessageId:o}=f();return(0,r.jsx)(i.g7,{ref:t,id:a,"aria-describedby":s?"".concat(l," ").concat(o):"".concat(l),"aria-invalid":!!s,...n})});h.displayName="FormControl";let j=s.forwardRef((e,t)=>{let{className:n,...s}=e,{formDescriptionId:i}=f();return(0,r.jsx)("div",{ref:t,id:i,className:(0,l.cn)("text-sm text-muted-foreground",n),...s})});j.displayName="FormDescription";let v=s.forwardRef((e,t)=>{let{className:n,children:s,...i}=e,{error:a,formMessageId:o}=f(),c=a?String(null==a?void 0:a.message):s;return c?(0,r.jsx)("p",{ref:t,id:o,className:(0,l.cn)("text-sm font-medium text-destructive",n),...i,children:c}):null});v.displayName="FormMessage"},5266:function(e,t,n){"use strict";n.d(t,{_:function(){return c}});var r=n(36164),s=n(3546),i=n(90893),a=n(14375),l=n(74248);let o=(0,a.j)("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"),c=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)(i.f,{ref:t,className:(0,l.cn)(o(),n),...s})});c.displayName=i.f.displayName},3448:function(e,t,n){"use strict";n.d(t,{O:function(){return i}});var r=n(36164),s=n(74248);function i(e){let{className:t,...n}=e;return(0,r.jsx)("div",{className:(0,s.cn)("h-4 animate-pulse rounded-md bg-gray-200 dark:bg-gray-700",t),...n})}},99047:function(e,t,n){"use strict";n.d(t,{RM:function(){return o},SC:function(){return u},iA:function(){return a},pj:function(){return f},ss:function(){return d},xD:function(){return l}});var r=n(36164),s=n(3546),i=n(74248);let a=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("table",{ref:t,className:(0,i.cn)("w-full caption-bottom text-sm",n),...s})});a.displayName="Table";let l=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("thead",{ref:t,className:(0,i.cn)("[&_tr]:border-b",n),...s})});l.displayName="TableHeader";let o=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("tbody",{ref:t,className:(0,i.cn)("[&_tr:last-child]:border-0",n),...s})});o.displayName="TableBody";let c=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("tfoot",{ref:t,className:(0,i.cn)("border-t bg-muted/50 font-medium [&>tr]:last:border-b-0",n),...s})});c.displayName="TableFooter";let u=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("tr",{ref:t,className:(0,i.cn)("border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted",n),...s})});u.displayName="TableRow";let d=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("th",{ref:t,className:(0,i.cn)("h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0",n),...s})});d.displayName="TableHead";let f=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("td",{ref:t,className:(0,i.cn)("p-4 align-middle [&:has([role=checkbox])]:pr-0",n),...s})});f.displayName="TableCell";let m=s.forwardRef((e,t)=>{let{className:n,...s}=e;return(0,r.jsx)("caption",{ref:t,className:(0,i.cn)("mt-4 text-sm text-muted-foreground",n),...s})});m.displayName="TableCaption"},29:function(e,t,n){"use strict";n.d(t,{_v:function(){return u},aJ:function(){return c},pn:function(){return l},u:function(){return o}});var r=n(36164),s=n(3546),i=n(44421),a=n(74248);let l=i.zt,o=i.fC,c=i.xz,u=s.forwardRef((e,t)=>{let{className:n,sideOffset:s=4,...l}=e;return(0,r.jsx)(i.VY,{ref:t,sideOffset:s,className:(0,a.cn)("z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-xs font-medium text-popover-foreground shadow-md animate-in fade-in-50 data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1",n),...l})});u.displayName=i.VY.displayName},24449:function(e,t,n){"use strict";n.d(t,{S:function(){return l},n:function(){return o}});var r=n(3546),s=n(45391),i=n(16784);let a=e=>{let t=(0,i.d)(e);r.useEffect(()=>()=>{t.current()},[])};function l(e,t,n){let l=(0,i.d)(e),o=r.useMemo(()=>(0,s.Z)(function(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];return l.current(...t)},t,n),[]);return a(()=>o.cancel()),{run:o,cancel:o.cancel,flush:o.flush}}function o(e,t,n){let[s,i]=r.useState(e),{run:a}=l(()=>{i(e)},t,n);return r.useEffect(()=>{a()},[e]),[s,i]}},16784:function(e,t,n){"use strict";n.d(t,{d:function(){return s}});var r=n(3546);function s(e){let t=r.useRef(e);return t.current=e,t}},29917:function(e,t,n){"use strict";n.d(t,{Gm:function(){return l},jp:function(){return a}});var r=n(40055),s=n(43240);let i=(0,s.BX)("\n  query GetLicenseInfo {\n    license {\n      type\n      status\n      seats\n      seatsUsed\n      issuedAt\n      expiresAt\n    }\n  }\n"),a=()=>(0,r.aM)({query:i}),l=()=>{let[{data:e}]=a();return null==e?void 0:e.license}},7600:function(e,t,n){"use strict";var r=n(48717).Z.Symbol;t.Z=r},64143:function(e,t){"use strict";t.Z=function(e,t){for(var n=-1,r=null==e?0:e.length,s=Array(r);++n<r;)s[n]=t(e[n],n,e);return s}},17996:function(e,t,n){"use strict";n.d(t,{Z:function(){return d}});var r=n(7600),s=Object.prototype,i=s.hasOwnProperty,a=s.toString,l=r.Z?r.Z.toStringTag:void 0,o=function(e){var t=i.call(e,l),n=e[l];try{e[l]=void 0;var r=!0}catch(e){}var s=a.call(e);return r&&(t?e[l]=n:delete e[l]),s},c=Object.prototype.toString,u=r.Z?r.Z.toStringTag:void 0,d=function(e){return null==e?void 0===e?"[object Undefined]":"[object Null]":u&&u in Object(e)?o(e):c.call(e)}},1282:function(e,t){"use strict";t.Z=function(e,t,n){var r=-1,s=e.length;t<0&&(t=-t>s?0:s+t),(n=n>s?s:n)<0&&(n+=s),s=t>n?0:n-t>>>0,t>>>=0;for(var i=Array(s);++r<s;)i[r]=e[r+t];return i}},4109:function(e,t,n){"use strict";var r=n(7600),s=n(64143),i=n(38813),a=n(55357),l=1/0,o=r.Z?r.Z.prototype:void 0,c=o?o.toString:void 0;t.Z=function e(t){if("string"==typeof t)return t;if((0,i.Z)(t))return(0,s.Z)(t,e)+"";if((0,a.Z)(t))return c?c.call(t):"";var n=t+"";return"0"==n&&1/t==-l?"-0":n}},77934:function(e,t,n){"use strict";var r=n(1282);t.Z=function(e,t,n){var s=e.length;return n=void 0===n?s:n,!t&&n>=s?e:(0,r.Z)(e,t,n)}},64380:function(e,t){"use strict";var n="object"==typeof global&&global&&global.Object===Object&&global;t.Z=n},59883:function(e,t){"use strict";var n=RegExp("[\\u200d\ud800-\udfff\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff\\ufe0e\\ufe0f]");t.Z=function(e){return n.test(e)}},48717:function(e,t,n){"use strict";var r=n(64380),s="object"==typeof self&&self&&self.Object===Object&&self,i=r.Z||s||Function("return this")();t.Z=i},14955:function(e,t,n){"use strict";n.d(t,{Z:function(){return x}});var r=n(59883),s="\ud800-\udfff",i="[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]",a="\ud83c[\udffb-\udfff]",l="[^"+s+"]",o="(?:\ud83c[\udde6-\uddff]){2}",c="[\ud800-\udbff][\udc00-\udfff]",u="(?:"+i+"|"+a+")?",d="[\\ufe0e\\ufe0f]?",f="(?:\\u200d(?:"+[l,o,c].join("|")+")"+d+u+")*",m=RegExp(a+"(?="+a+")|(?:"+[l+i+"?",i,o,c,"["+s+"]"].join("|")+")"+(d+u+f),"g"),x=function(e){return(0,r.Z)(e)?e.match(m)||[]:e.split("")}},88542:function(e,t,n){"use strict";n.d(t,{Z:function(){return o}});var r=n(53294),s=n(77934),i=n(59883),a=n(14955),l=function(e){e=(0,r.Z)(e);var t=(0,i.Z)(e)?(0,a.Z)(e):void 0,n=t?t[0]:e.charAt(0),l=t?(0,s.Z)(t,1).join(""):e.slice(1);return n.toUpperCase()+l},o=function(e){return l((0,r.Z)(e).toLowerCase())}},38813:function(e,t){"use strict";var n=Array.isArray;t.Z=n},84639:function(e,t){"use strict";t.Z=function(e){var t=typeof e;return null!=e&&("object"==t||"function"==t)}},96786:function(e,t){"use strict";t.Z=function(e){return null!=e&&"object"==typeof e}},53294:function(e,t,n){"use strict";var r=n(4109);t.Z=function(e){return null==e?"":(0,r.Z)(e)}}},function(e){e.O(0,[7565,7998,5498,6312,4007,2134,6201,3449,2578,5448,8511,2672,3882,3894,7444,1565,1624,3396,3375,5289,1744],function(){return e(e.s=47714)}),_N_E=e.O()}]);