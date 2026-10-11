const rmCheck = document.getElementById("ckb1");
const usernameInput = document.getElementById("usrInput1");

if (localStorage.checkbox && localStorage.checkbox !== "") {
  rmCheck.setAttribute("checked", "checked");
  usernameInput.value = localStorage.username;
} else {
  rmCheck.removeAttribute("checked");
  usernameInput.value = "";
}

function lsRememberMe() {
  if (rmCheck.checked && usernameInput.value !== "") {
    localStorage.username = usernameInput.value;
    localStorage.checkbox = rmCheck.value;
  } else {
    localStorage.username = "";
    localStorage.checkbox = "";
  }
}