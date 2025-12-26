let demographicChartInstance
let animeTimeChartInstance

// fungsi input username
const inputUsername = (ttlMs) => {
    Swal.fire({
        title: "Submit your MAL username",
        input: "text",
        inputAttributes: {
            autocapitalize: "off"
        },
        confirmButtonText: "Submit",
        showLoaderOnConfirm: true,
        allowOutsideClick: false,
        allowEscapeKey: false,
        showCloseButton: false,
        showCancelButton: false,
        preConfirm: async (username) => {
            try {
                if (!username || username.trim() == "") {
                    Swal.showValidationMessage("Please enter username!");
                    return false;
                }

                try {
                    const response = await startScraping(username)
                    return response
                } catch (error) {
                    Swal.showValidationMessage(`Request failed: ${error}`);
                    return false;
                }
                
            } catch (error) {
                Swal.showValidationMessage(`
                    Request failed: ${error}
                `);
            }
        }
    })
    .then(async (result) => {
        if (result.isConfirmed) {
            animelist = result.value
            setSessionStorage('animelist', animelist, ttlMs) // simpan di session storage selama 60 menit

            const analysis = await startAnalysis(animelist.data)
            await chartManagement(animelist, analysis)

            initialFilter(analysis)
            
            const recomendAnime = await recommendationAnime(animelist.data)
            initialRecommendation(recomendAnime)
        }
    });
}

// fungsi untuk session storage dengan expiry
const getSessionStorage = (key) => {
    const itemStr = sessionStorage.getItem(key)
    if (!itemStr) return null

    const item = JSON.parse(itemStr)
    const now = Date.now()

    if (now > item.expiry) {
        sessionStorage.removeItem(key)
        return null
    }

    return item.value
};

// fungsi untuk session storage dengan expiry
const setSessionStorage = (key, value, ttlMs) => {
    const now = Date.now()

    const item = {
        value: value,
        expiry: now + ttlMs
    }

    sessionStorage.setItem(key, JSON.stringify(item))
};

const removeSessionStorage = (key) => {
    sessionStorage.removeItem(key)
}

// fungsi untuk memulai scraping
const startScraping = async (username_) => {           
    try {
        const response = await fetch(`/scrape?username=${username_}`)
        return response.json();
    } catch (error) {
        Swal.fire({
            icon: 'error',
            title: 'Sorry Something Went Wrong',
            text: `
                Hmmmm... we encountered an error during the scraping process. Please try again later. \n
                Are you sure the username "${username_}" exists?
            `,
        });
    }
}

const refreshScraping = async (username_) => {
    Swal.fire({
            title: 'Are you sure?',
            text: "This will delete your previous data and start a new scraping process.",
            icon: 'warning',
            showCancelButton: true,
            confirmButtonColor: '#ffb86a',
            cancelButtonColor: '#d33',
            confirmButtonText: 'Yes, refresh it!'
        }).then((result) => {
            if (result.isConfirmed) {
                removeSessionStorage('animelist');
                inputUsername(15 * 60 * 1000); // 5 menit
            }
        });
}

// fungsi untuk memulai analysis
const startAnalysis = async (data) => {
    try {        
        const response = await fetch(`/analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        return response.json();
    } catch (error) {
        Swal.fire({
            icon: 'error',
            title: 'Sorry Something Went Wrong',
            text: `Hmmmm... we encountered an error during the analysis process. Please try again later.`,
        });
    }       
}

const recommendationAnime = async (data) => {
    try {
        const response = await fetch(`/recommendation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
    
        return response.json();
    } catch (error) {
        Swal.fire({
            icon: 'error',
            title: 'Sorry Something Went Wrong',
            text: `Hmmmm... we encountered an error during the analysis process. Please try again later.`,
        });
    }
}

const initialRecommendation = (data) =>{
    const animerecommend_ = document.getElementById('animerecommend_')

    let html = ``
    data.map((item, i) => {
        html += `
            <a href="https://myanimelist.net/anime/${item.mal_id}" target="_blank" class="flex flex-1 gap-2 py-1 rounded-xl px-2 cursor-pointer transition duration-300 ease-in-out hover:bg-gray-200 hover:scale-105 group">
                <img src="${item.image_url}" class="h-15 aspect-square rounded-2xl">
                <div class="flex justify-between items-center w-full">
                    <div>
                        <p class="font-bold  group-hover:text-gray-600">${item.title.slice(0, 20) + (item.title.length > 15 ? "..." : "")}</p>
                        <p class="text-sm">Total Episodes: <strong>${item.episode}</strong></p>
                        <p class="text-sm">Similarity: <strong>${(item.similarity_score).toFixed(2)}</strong></p>
                    </div>
                    
                    <div class="flex justify-center items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" class="text-yellow-500" viewBox="0 0 16 16">
                            <path d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
                        </svg>
                        <p class="font-bold">${item.mal_score}</p>
                    </div>
                </div>
            </a>
        `
    })

    animerecommend_.innerHTML = html
}

// fungsi untuk mengelola chart dan data
const chartManagement = async (response, analysis) => {
    // set anime list, watch, and unwatch
    const watched_ = animeWatched(response)
    const {
        genre, 
        studio, 
        producer, 
        demographic, 
        theme, 
        anime_time
    } = analysis

    const topGenre = [...genre].sort((a, b) => b[1] - a[1]).slice(0, 5)
    const topStudio = [...studio].sort((a, b) => b[1] - a[1]).slice(0, 5)
    const topProducer = [...producer].sort((a, b) => b[1] - a[1]).slice(0, 5)

    genreWatched(topGenre)
    studioWatched(topStudio)
    producerWatched(topProducer)

    demographicChart(demographic)
    animeTimeChart(anime_time)
}

// fungsi untuk menampilkan anime watched dan unwatched
const animeWatched = (response) => {
    const animewatched_ = document.getElementById('animewatched_')
    const stat_watched = document.getElementById('stat_watched')
    const stat_unwatched = document.getElementById('stat_unwatched')
    let watched_count = 0
    let unwatched_count = 0

    const checkProgress = (string) => {        
        let progress = string.split('/')
        if(progress[1]) {
            unwatched_count += 1
            return `${progress[0]}/${progress[1]}`
        }

        watched_count += 1
        return `<div class="flex flex-1 gap-2"><p>${progress[0]}</p> <p class="text-green-500 font-bold">Done!</p></div>`
    }
    
    let html = ``
    response.data.map((item, i) => {
        html += `
            <a href="https://myanimelist.net/anime/${item.id}" target="_blank" class="flex flex-1 gap-2 py-1 rounded-xl px-2 cursor-pointer transition duration-300 ease-in-out hover:bg-gray-200 hover:scale-105 group">
                <img src="${item.image}" class="h-15 aspect-square rounded-2xl">
                <div class="flex justify-between items-center w-full">
                    <div>
                        <p class="font-bold  group-hover:text-gray-600">${item.title.slice(0, 20) + (item.title.length > 15 ? "..." : "")}</p>
                        <p class="text-sm">${checkProgress(item.Progress)}</p>
                    </div>
                    
                    <div class="flex justify-center items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" class="text-yellow-500" viewBox="0 0 16 16">
                            <path d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
                        </svg>
                        <p class="font-bold">${item.score}</p>
                    </div>
                </div>
            </a>
        `
    })
    animewatched_.innerHTML = html

    stat_watched.innerText = watched_count
    stat_unwatched.innerText = unwatched_count
}

// fungsi untuk menampilkan top genre, studio, dan producer
const genreWatched = (data) => {
    const genrewatched_ = document.getElementById('genrewatched_')
    let html = ``

    data.map((item, i) => {
        html += `
            <div class="flex justify-between items-center mb-1 px-3 rounded-md cursor-pointer transition duration-300 ease-in-out hover:bg-gray-200 hover:scale-105 group">
                <p class="group-hover:text-gray-600">${item.genres}</p>
                <p class="font-bold">${item.count}</p>
            </div>
        `
    })
    genrewatched_.innerHTML = html
}

// fungsi untuk menampilkan top studio
const studioWatched = (data) => {
    const studiowatched_ = document.getElementById('studiowatched_')
    let html = ``

    data.map((item, i) => {
        html += `
            <div class="flex justify-between items-center mb-1 px-3 rounded-md cursor-pointer transition duration-300 ease-in-out hover:bg-gray-200 hover:scale-105 group">
                <p class="group-hover:text-gray-600">${item.studios}</p>
                <p class="font-bold">${item.count}</p>
            </div>
        `
    })
    studiowatched_.innerHTML = html
}

// fungsi untuk menampilkan top producer
const producerWatched = (data) => {
    const procedurwatched_ = document.getElementById('procedurwatched_')
    let html = ``

    data.map((item, i) => {
        html += `
            <div class="flex justify-between items-center mb-1 px-3 rounded-md cursor-pointer transition duration-300 ease-in-out hover:bg-gray-200 hover:scale-105 group">
                <p class="group-hover:text-gray-600">${item.producers}</p>
                <p class="font-bold">${item.count}</p>
            </div>
        `
    })
    procedurwatched_.innerHTML = html
}

// fungsi untuk menampilkan chart demographic dan anime time
const demographicChart = (data) => {
    const ctx = document.getElementById('demographicchart_')

    if (demographicChartInstance && typeof demographicChartInstance.destroy === 'function') {
        demographicChartInstance.destroy()
        demographicChartInstance = null;
    }

    demographicChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.map(item => item.demographics),
            datasets: [{
                data: data.map(item => item.count),
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            elements: {
                line: {
                    borderWidth: 3
                }
            }
        }
    })
}

// fungsi untuk menampilkan chart anime time
const animeTimeChart = (data) => {
    const ctx = document.getElementById('animetimechart_')

    if (animeTimeChartInstance && typeof animeTimeChartInstance.destroy === 'function') {
        animeTimeChartInstance.destroy()
        animeTimeChartInstance = null;
    }

    animeTimeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(item => item.time),
            datasets: [{
                label: 'Anime Released Over Time',
                data: data.map(item => item.count),
                fill: false,
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (Year Season)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Number of Anime Released'
                    },
                    beginAtZero: true
                }
            }
        }
    })
}

// fungsi untuk inisialisasi filter
const initialFilter = (data) => {
    const filterGenre = document.getElementById('genreFilter');
    const filterStudio = document.getElementById('StudioFilter');

    const genreAll = document.getElementById('genreAll');
    const studioAll = document.getElementById('studioAll');

    const genreSearch = document.getElementById('genreSearch');
    const studioSearch = document.getElementById('studioSearch');

    data.genre.forEach(item => {
        const option = createOption(item.genres, item.count, 'genre');
        filterGenre.appendChild(option);
    });
    data.studio.forEach(item => {
        const option = createOption(item.studios, item.count, 'studio');
        filterStudio.appendChild(option);
    });

    genreAll.checked = true;
    studioAll.checked = true;
    toggleAll(filterGenre, true);
    toggleAll(filterStudio, true);

    genreAll.addEventListener('change', () => {
        toggleAll(filterGenre, genreAll.checked);
    });

    studioAll.addEventListener('change', () => {
        toggleAll(filterStudio, studioAll.checked);
    });
    genreSearch.addEventListener('input', () => {
        searchFilter(filterGenre, genreSearch.value);
    });

    studioSearch.addEventListener('input', () => {
        searchFilter(filterStudio, studioSearch.value);
    });
};

// fungsi pembantu untuk filter
const createOption = (value, count, type) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'flex items-center mb-2 filter-item hover:bg-gray-200 px-2 rounded-md cursor-pointer transition duration-300 ease-in-out';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = value;
    checkbox.className = `${type}-checkbox mr-2`;
    checkbox.id = `${type}-${value.replace(/\s+/g, '-')}`;
    checkbox.onchange = () => {
        handleSingleCheck(type);
    };

    const label = document.createElement('label');
    label.textContent = `${value} (${count})`;
    label.className = 'cursor-pointer w-full';
    label.htmlFor = checkbox.id;

    wrapper.appendChild(checkbox);
    wrapper.appendChild(label);

    return wrapper;
};

// fungsi pembantu untuk filter
const toggleAll = (container, checked) => {
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = checked);
};

// fungsi pembantu untuk filter
const searchFilter = (container, keyword) => {
    const items = container.querySelectorAll('.filter-item');
    const search = keyword.toLowerCase();

    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(search) ? 'flex' : 'none';
    });
};

// fungsi pembantu untuk filter
const handleSingleCheck = (type) => {
    const allCheckbox = document.getElementById(type + 'All');
    const container = document.getElementById(
        type === 'genre' ? 'genreFilter' : 'StudioFilter'
    );

    const checkboxes = container.querySelectorAll(`.${type}-checkbox`);
    const checked = [...checkboxes].filter(cb => cb.checked);

    // Jika ada yang tidak dicentang → All off
    if (checked.length !== checkboxes.length) {
        allCheckbox.checked = false;
    }

    // Jika semua dicentang → All on
    if (checked.length === checkboxes.length) {
        allCheckbox.checked = true;
    }
}

// fungsi untuk action change filter
const applyFilters = async () => {
    const genreCheckboxes = document.querySelectorAll('.genre-checkbox');
    const studioCheckboxes = document.querySelectorAll('.studio-checkbox');

    const checkedGenres = [...genreCheckboxes]
        .filter(cb => cb.checked)
        .map(cb => cb.value.trim());

    const checkedStudios = [...studioCheckboxes]
        .filter(cb => cb.checked)
        .map(cb => cb.value.trim());
    try {        
        const response = await fetch('/filter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selected_genres: checkedGenres,
                selected_studios: checkedStudios,
                data: getSessionStorage('animelist').data
            })
        })
    
        const result = await response.json();
    
        if (!response.ok){
            throw new Error(JSON.stringify(await response.json()));
        }

        const setList = {
            username: getSessionStorage('animelist').username,
            total_items: result.total_items,
            data: result.data
        }

        const analysis = await startAnalysis(setList.data)
        await chartManagement(setList, analysis)

        const recomendAnime = await recommendationAnime(setList.data)
        initialRecommendation(recomendAnime)

    } catch (error) {
        console.log(error);
        Swal.fire({
            icon: 'error',
            title: 'Sorry Something Went Wrong',
            text: `Hmmmm... we encountered an error during the filtering process. Please try again later.`,
        });
    }
}

// Filter Watch & Unwatch
const filterWatchStatus = async (status) => {
    const animelist = sessionStorage.getItem('animelist');
    if (!animelist) return;
    
    const parsedList = JSON.parse(animelist).value;
    const animeData = parsedList.data;

    const filteredAnime = animeData.filter(({ Progress }) => {
        const [, total] = Progress.split('/');
        return (
            (status === 'unwatched' && total) ||
            (status === 'watched' && !total)
        );
    }); 

    const analysis = await startAnalysis(filteredAnime)
    await chartManagement({ ...parsedList, data: filteredAnime }, analysis)
    initialFilter(analysis)

    const recomendAnime = await recommendationAnime(filteredAnime)
    initialRecommendation(recomendAnime)
}