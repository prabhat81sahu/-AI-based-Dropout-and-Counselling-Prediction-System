// simple script to fetch recommendations based on current form values
// and display them below the form without submitting

async function fetchRecommendations() {
    const grades = document.querySelector('input[name="grades"]').value;
    const gpa = document.querySelector('input[name="gpa"]').value;
    const attendance = document.querySelector('input[name="attendance"]').value;
    const behavior = document.querySelector('select[name="behavior"]').value;
    const socio = document.querySelector('select[name="socio_eco"]').value;

    // only attempt when enough data present
    if (!grades || !gpa || !attendance) {
        return;
    }

    const response = await fetch('/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            risk: 'Medium', // placeholder; server doesn't require risk for rule-based logic
            behavior: behavior
        })
    });
    const data = await response.json();
    const list = document.getElementById('rec-list');
    list.innerHTML = '';
    data.recommendations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        list.appendChild(li);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(el => {
        el.addEventListener('change', fetchRecommendations);
    });
});
