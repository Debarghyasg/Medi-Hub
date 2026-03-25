 function openModal() {
    document.getElementById('modalOverlay').classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    document.body.style.overflow = '';
    document.getElementById('file-preview').style.display = 'none';
    document.getElementById('file-preview').textContent = '';
  }

  function closeOnOverlay(e) {
    if (e.target === document.getElementById('modalOverlay')) closeModal();
  }

  function handleFile(e) {
    const file = e.target.files[0];
    if (file) {
      const preview = document.getElementById('file-preview');
      preview.style.display = 'block';
      preview.textContent = `✅  Selected: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    }
  }

  function submitUpload() {
    const preview = document.getElementById('file-preview');
    if (!preview.textContent) {
      preview.style.display = 'block';
      preview.style.background = '#fff0f0';
      preview.style.color = '#c0392b';
      preview.textContent = '⚠️  Please select a file first.';
      return;
    }
    preview.style.background = 'var(--mint)';
    preview.style.color = 'var(--teal-deep)';
    preview.textContent = '🎉  Prescription uploaded successfully!';
    setTimeout(closeModal, 1500);
  }

  // Drag & Drop
  const dz = document.getElementById('dropZone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      const preview = document.getElementById('file-preview');
      preview.style.display = 'block';
      preview.textContent = `✅  Selected: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    }
  });

  // ESC to close
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });



  // Add this inside your script or /app.js
async function submitUpload() {
    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];
    if (!file) return alert("Please select a file first.");

    const formData = new FormData();
    formData.append('prescription', file);

    // Show a loading state on the button
    const btn = document.querySelector('.btn-upload-confirm');
    btn.innerText = "Processing...";
    btn.disabled = true;

    try {
        const response = await fetch('/upload', { // Your axios route
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
       if (result.success) {
    // We use the redirectUrl sent back by your index.js 
    // It will look like "/confirm/123"
    window.location.href = result.redirectUrl;
} else {
    alert("Extraction failed: " + (result.message || "Unknown error"));
}
    } catch (err) {
        console.error(err);
        alert("Extraction failed. Please try again.");
    } finally {
        btn.innerText = "Upload & Save";
        btn.disabled = false;
    }
}
