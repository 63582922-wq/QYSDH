/**
 * 所有 createPortal 统一挂到此节点，避免与 #root 并列插在 document.body 上时，
 * 在部分 WebKit / 多 Portal 同帧更新场景下触发 insertBefore 与 DOM 不一致。
 */
export function getAppPortalNode(): HTMLElement {
  let el = document.getElementById('app-portals');
  if (el) return el;
  el = document.createElement('div');
  el.id = 'app-portals';
  el.setAttribute('data-react-app-portals', '');
  document.body.appendChild(el);
  return el;
}
