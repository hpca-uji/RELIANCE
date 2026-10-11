// Prevent Double Submits
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', (e) => {
        // Prevent if already submitting
        if (form.classList.contains('is-submitting')) {
            e.preventDefault();
        }

        // Add a visual indicator to show the user it is submitting
        form.classList.add('is-submitting');
    });
});