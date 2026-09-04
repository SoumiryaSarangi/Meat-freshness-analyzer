// shared.js -- pure logic, no DOM references
export async function classifyImages(files) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await fetch('/api/classify', { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`Server responded ${res.status}`);
  return (await res.json()).results;
}

export function downloadAnnotatedImage(base64DataUrl, filename) {
  const b64 = base64DataUrl.replace('data:image/jpeg;base64,', '');
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  const blob = new Blob([arr], { type: 'image/jpeg' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `annotated_${filename.replace(/\.[^.]+$/, '')}.jpg`;
  a.click();
  URL.revokeObjectURL(a.href);
}
