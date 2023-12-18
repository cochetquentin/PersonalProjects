// Sélectionnez les éléments du DOM
const texteATraduireInput = document.getElementById("texteATraduire");
const boutonTraduction = document.getElementById("boutonTraduction");
const traductionTextarea = document.getElementById("traduction");
const explicationsTextarea = document.getElementById("explications");
const spinner = document.getElementById("spinner"); // Sélectionnez l'élément du spinner

// Fonction pour ajuster la hauteur de la boîte de dialogue d'explications
function ajusterHauteurExplicationsTextarea() {
    // Réinitialisez la hauteur de la boîte de dialogue à 0
    explicationsTextarea.style.height = "auto";

    // Définissez la hauteur de la boîte de dialogue en fonction de son contenu
    explicationsTextarea.style.height = (explicationsTextarea.scrollHeight + 10) + "px";
}

// Fonction pour afficher le spinner
function afficherSpinner() {
    spinner.style.display = "block";
}

// Fonction pour masquer le spinner
function masquerSpinner() {
    spinner.style.display = "none";
}

// Fonction pour effectuer la traduction
function effectuerTraduction() {
    // Affichez le spinner au début du chargement
    afficherSpinner();

    const texteATraduire = texteATraduireInput.value;

    // Effectuez la requête API pour obtenir la traduction
    fetch('/api/traduction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: texteATraduire })
    })
    .then(response => response.json())
    .then(data => {
        const traduction = data.translation;
        const explications = data.explanations;

        // Mettez à jour les boîtes de dialogue de traduction et d'explications
        traductionTextarea.value = traduction;
        explicationsTextarea.value = explications;

        // Ajustez la hauteur de la boîte de dialogue d'explications
        ajusterHauteurExplicationsTextarea();

        // Masquez le spinner lorsque le chargement est terminé
        masquerSpinner();
    })
    .catch(error => {
        console.error('Erreur :', error);
        // Masquez le spinner en cas d'erreur
        masquerSpinner();
    });
}

// Écoutez le clic sur le bouton de traduction
boutonTraduction.addEventListener("click", effectuerTraduction);

// Appelez la fonction pour masquer le spinner initialement
masquerSpinner();
