// Extracts parameters from URL and initializes global variables
const params = new URLSearchParams(window.location.search);
const tournamentValue = params.get('tournament') || 'default';
const competitionSelect = document.getElementById('competition-select');
const refreshButton = document.getElementById('refresh_button');
const spinner = document.getElementById('spinner');
const plots = document.querySelectorAll('.plot');
const statsCheckboxes = document.querySelectorAll('.checkbox_stats');
const teamCheckboxes = document.querySelectorAll('.checkbox_teams');
const teamPlots = document.querySelectorAll('.by_team');

// Initializes the page based on URL parameters
function initializePage() {
    competitionSelect.value = tournamentValue;
    // Displays the refresh button if a specific tournament is selected
    refreshButton.style.display = tournamentValue !== 'default' ? 'block' : 'none';
}

// Displays spinner and hides all plot elements
function displaySpinnerAndHidePlots() {
    spinner.style.display = 'block';
    plots.forEach(plot => plot.style.display = 'none');
}

// Redirects to the same page with updated query parameters
function redirectToUpdatedUrl(tournament, refresh = false) {
    let newUrl = window.location.pathname + '?tournament=' + tournament;
    if (refresh) newUrl += '&refresh=true';
    window.location.href = newUrl;
}

// Function to update the display of images based on selected filters
function updateImageDisplay() {
    // Array to store checked statistics options
    let checkedStats = [];
    // Collects all checked statistics checkboxes and updates checkedStats array
    statsCheckboxes.forEach(function(checkbox) {
        if(checkbox.checked) {
            checkedStats.push(checkbox.value);
        }
    });

    // Toggles display of plot images based on checked statistics
    plots.forEach(function(img){
        img.style.display = "none"; // Initially hides all images
        checkedStats.forEach(function(stat){
            if(img.classList.contains(stat)) {
                img.style.display = "block"; // Shows images that match checked statistics
            }
        });
    });

    // Array to store checked teams options
    let checkedTeams = [];
    // Collects all checked teams checkboxes and updates checkedTeams array
    teamCheckboxes.forEach(function(checkbox) {
        if(checkbox.checked) {
            checkedTeams.push(checkbox.value);
        }
    });

    // Filters and displays images by team selection
    const displayedImages = Array.from(teamPlots).filter(img => img.style.display !== "none");

    displayedImages.forEach(function(img){
        img.style.display = "none"; // Initially hides images
        checkedTeams.forEach(function(team){
            let breakFlag = false;
            team.split(' ').forEach(function(t){
                if(!img.classList.contains(t)) {
                    breakFlag = true; // Sets flag if image doesn't match team criteria
                }         
            });
            if(!breakFlag) {
                img.style.display = "block"; // Shows image if it matches all team criteria
            }
        });
    });
}

function setupEventListeners() {
    // Event listener for competition select dropdown
    competitionSelect.addEventListener('change', function() {
        displaySpinnerAndHidePlots();
        redirectToUpdatedUrl(this.value);
    });

    // Event listener for the refresh button
    refreshButton.addEventListener('click', function() {
        displaySpinnerAndHidePlots();
        redirectToUpdatedUrl(tournamentValue, true);
    });

    // Adds change event listeners to all statistics and teams checkboxes
    document.querySelectorAll('.checkbox_stats, .checkbox_teams').forEach(function(checkbox) {
        checkbox.addEventListener('change', updateImageDisplay);
    });
}

// Initialization function
function initialize() {
    initializePage();
    setupEventListeners();
    updateImageDisplay(); // Apply initial filter settings on load
}

// Call initialize on window load
window.onload = initialize;