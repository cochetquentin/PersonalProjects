// Display spinner and make get request
document.getElementById('competition-select').addEventListener('change', function() {
    document.getElementById('spinner').style.display = 'block';
    window.location.href = window.location.pathname + '?tournament=' + this.value;
});

// Fonction pour mettre à jour l'affichage des images
function updateImageDisplay() {
    let checkedStats = [];
    document.querySelectorAll(".checkbox_stats").forEach(function(checkbox) {
        if(checkbox.checked) {
            checkedStats.push(checkbox.value);
        }
    });

    document.querySelectorAll(".plot").forEach(function(img){
        img.style.display = "none";
        checkedStats.forEach(function(stat){
            if(img.classList.contains(stat)) {
                img.style.display = "block";
            }
        });
    });


    let checkedTeams = [];
    document.querySelectorAll(".checkbox_teams").forEach(function(checkbox) {
        if(checkbox.checked) {
            checkedTeams.push(checkbox.value);
        }
    });

    const images = document.querySelectorAll('.by_team');
    const displayedImages = Array.from(images).filter(img => {
        return img.style.display !== "none";
    });

    displayedImages.forEach(function(img){
        img.style.display = "none";
        checkedTeams.forEach(function(team){
            let breakFlag = false;
            team.split(' ').forEach(function(t){
                if(!img.classList.contains(t)) {
                    breakFlag = true;
                }         
            });
            if(!breakFlag) {
                img.style.display = "block";
            }
        });
    });
}


// Ajouter un écouteur d'événement sur les checkboxes
document.querySelectorAll('.checkbox_stats').forEach(function(checkbox) {
    checkbox.addEventListener('change', updateImageDisplay);
});
document.querySelectorAll('.checkbox_teams').forEach(function(checkbox) {
    checkbox.addEventListener('change', updateImageDisplay);
});

// Mise à jour initiale pour afficher les images correctes au chargement de la page
window.onload = updateImageDisplay;