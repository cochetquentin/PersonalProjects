
document.getElementById('competition-select').addEventListener('change', function() {
    var selectedValue = this.value;
    if(selectedValue) {
        // Construit l'URL basé sur la sélection et redirige l'utilisateur
        window.location = `./${selectedValue}`;
    }
});

// Fonction pour mettre à jour l'affichage des images
function updateImageDisplay() {
    document.querySelectorAll('.plot').forEach(function(img) {
        // Initialiser une variable pour vérifier si l'image doit être affichée
        let isImageVisible = false;
        
        // Obtenir toutes les checkboxes
        const checkboxes = document.querySelectorAll('.checkbox');
        
        // Itérer sur chaque checkbox pour vérifier si au moins une est cochée et correspond à une classe de l'image
        checkboxes.forEach(function(checkbox) {
            if(checkbox.checked && img.classList.contains(checkbox.value)) {
                isImageVisible = true;
            }
        });

        // Afficher ou cacher l'image basé sur la variable isImageVisible
        img.style.display = isImageVisible ? 'block' : 'none';
    });
}

// Ajouter un écouteur d'événement sur les checkboxes
document.querySelectorAll('.checkbox').forEach(function(checkbox) {
    checkbox.addEventListener('change', updateImageDisplay);
});

// Mise à jour initiale pour afficher les images correctes au chargement de la page
window.onload = updateImageDisplay;
