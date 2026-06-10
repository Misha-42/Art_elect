const galleryItems = [
  {
    file: "../plots/fig_02_01_architecture.png",
    title: "Рисунок 2.1 — Трёхуровневая архитектура",
    note: "Полевой, сетевой и корпоративный уровни",
    groups: ["architecture"],
  },
  {
    file: "../plots/fig_02_02_modules.png",
    title: "Рисунок 2.2 — Функциональные модули",
    note: "Мониторинг, диагностика, прогноз и управление",
    groups: ["architecture"],
  },
  {
    file: "../plots/fig_02_03_zone_map.png",
    title: "Рисунок 2.3 — Карта зон сети",
    note: "Городская, промышленная и сельская зоны",
    groups: ["data"],
  },
  {
    file: "../plots/fig_02_04_information_flow.png",
    title: "Рисунок 2.4 — Информационные потоки",
    note: "От телеметрии до управляющих рекомендаций",
    groups: ["data"],
  },
  {
    file: "../plots/fig_02_05_sources.png",
    title: "Рисунок 2.5 — Источники данных",
    note: "Оперативная телеметрия (SCADA), АИИС КУЭ, паспорт оборудования и контекст",
    groups: ["data"],
  },
  {
    file: "../plots/fig_02_06_preprocessing.png",
    title: "Рисунок 2.6 — Предобработка данных",
    note: "Очистка, восстановление пропусков, валидация",
    groups: ["data"],
  },
  {
    file: "../plots/fig_02_07_features.png",
    title: "Рисунок 2.7 — Формирование признаков",
    note: "Лаги, окна, отношения и календарные признаки",
    groups: ["ml"],
  },
  {
    file: "../plots/report_table_1.png",
    title: "Таблица 1 — Сравнение алгоритмов",
    note: "Модель CatBoost (CatBoost), LSTM (LSTM) и линейная регрессия",
    groups: ["tables", "ml"],
  },
  {
    file: "../plots/fig_02_08_training.png",
    title: "Рисунок 2.8 — Обучение модели",
    note: "Подбор гиперпараметров и проверка качества",
    groups: ["ml"],
  },
  {
    file: "../plots/report_table_2.png",
    title: "Таблица 2 — Гиперпараметры по зонам",
    note: "Набор настроек для городской, промышленной и сельской сети",
    groups: ["tables", "ml"],
  },
  {
    file: "../plots/fig_02_09_metrics.png",
    title: "Рисунок 2.9 — Метрики оценки",
    note: "MAE, RMSE, R² и горизонт прогноза",
    groups: ["ml"],
  },
  {
    file: "../plots/report_table_3.png",
    title: "Таблица 3 — Ожидаемые метрики прогноза",
    note: "Диапазоны качества по типам зон",
    groups: ["tables", "ml"],
  },
  {
    file: "../plots/fig_02_10_sources_for_rules.png",
    title: "Рисунок 2.10 — Источники знаний",
    note: "Регламенты, архивы событий и экспертный опыт",
    groups: ["expert"],
  },
  {
    file: "../plots/report_table_4.png",
    title: "Таблица 4 — Примеры правил",
    note: "Продукционные правила для диспетчерского контура",
    groups: ["tables", "expert"],
  },
  {
    file: "../plots/fig_02_11_knowledge_base.png",
    title: "Рисунок 2.11 — База знаний",
    note: "Факты, правила и метазнания",
    groups: ["expert"],
  },
  {
    file: "../plots/report_table_5.png",
    title: "Таблица 5 — Структура базы знаний",
    note: "Факты, правила и метазнания в одном контуре",
    groups: ["tables", "expert"],
  },
  {
    file: "../plots/fig_02_12_hybrid_architecture.png",
    title: "Рисунок 2.12 — Гибридная архитектура",
    note: "Сценарии взаимодействия машинного обучения (ML) и экспертной системы",
    groups: ["expert"],
  },
  {
    file: "../plots/report_table_6.png",
    title: "Таблица 6 — Сравнение архитектур",
    note: "Гибридная схема против моно-подходов",
    groups: ["tables", "expert"],
  },
  {
    file: "../plots/fig_02_13_fault_localization.png",
    title: "Рисунок 2.13 — Локализация повреждений",
    note: "Телеметрия, кросс-верификация и карта аварии",
    groups: ["algorithms"],
  },
  {
    file: "../plots/report_table_7.png",
    title: "Таблица 7 — Показатели выявления повреждений",
    note: "Время обнаружения, точность и ложные срабатывания",
    groups: ["tables", "algorithms"],
  },
  {
    file: "../plots/fig_02_14_isolation_reconfiguration.png",
    title: "Рисунок 2.14 — Изоляция и реконфигурация",
    note: "Переход на резерв и восстановление питания",
    groups: ["algorithms"],
  },
  {
    file: "../plots/report_table_8.png",
    title: "Таблица 8 — Показатели реконфигурации",
    note: "Изоляция, восстановление и доля потребителей",
    groups: ["tables", "algorithms"],
  },
  {
    file: "../plots/fig_02_15_power_flow.png",
    title: "Рисунок 2.15 — Оптимизация потоков мощности",
    note: "Снижение потерь и перераспределение нагрузки",
    groups: ["algorithms"],
  },
  {
    file: "../plots/report_table_9.png",
    title: "Таблица 9 — Снижение потерь электроэнергии",
    note: "Ожидаемый экономический эффект по зонам",
    groups: ["tables", "algorithms"],
  },
  {
    file: "../plots/fig_02_perimeter_architecture.png",
    title: "Рисунок — Защищённый периметр",
    note: "Зоны и каналы (Zone & Conduit), демилитаризованная зона (DMZ), межсетевой экран нового поколения (NGFW) и журналирование",
    groups: ["security"],
  },
  {
    file: "../plots/fig_02_16_auth_audit.png",
    title: "Рисунок 2.16 — Аутентификация, авторизация и аудит",
    note: "Ролевая модель доступа (RBAC), многофакторная аутентификация (MFA) и контроль доступа",
    groups: ["security"],
  },
  {
    file: "../plots/fig_02_17_crypto_resilience.png",
    title: "Рисунок 2.17 — Криптозащита и отказоустойчивость",
    note: "Шифрование канала (TLS), контроль целостности и резервирование",
    groups: ["security"],
  },
  {
    file: "../plots/fig_02_18_testing.png",
    title: "Рисунок 2.18 — Тестирование компонентов",
    note: "От модульных тестов к цифровому двойнику",
    groups: ["testing"],
  },
  {
    file: "../plots/fig_02_19_validation.png",
    title: "Рисунок 2.19 — Валидация машинного обучения (ML) и экспертной системы",
    note: "Скользящая ретроспективная проверка (Walk-Forward), архив аварийных сценариев и стресс-тесты",
    groups: ["testing"],
  },
  {
    file: "../plots/fig_02_20_acceptance.png",
    title: "Рисунок 2.20 — Критерии приёмки и пилот",
    note: "Готовность, надёжность, киберустойчивость и экономика",
    groups: ["roadmap"],
  },
];

const galleryRoot = document.querySelector("[data-gallery]");
const countNode = document.querySelector("[data-gallery-count]");
const filterButtons = Array.from(document.querySelectorAll("[data-filter]"));

function renderGallery(activeFilter = "all") {
  const visibleItems = galleryItems.filter((item) => {
    if (activeFilter === "all") {
      return true;
    }

    return item.groups.includes(activeFilter);
  });

  countNode.textContent = String(visibleItems.length);

  galleryRoot.innerHTML = visibleItems
    .map(
      (item) => `
        <a class="gallery-card" href="${item.file}" target="_blank" rel="noreferrer" data-groups="${item.groups.join(",")}" data-reveal>
          <div class="gallery-card__media">
            <img src="${item.file}" alt="${item.title}" loading="lazy" decoding="async">
          </div>
          <div class="gallery-card__body">
            <span class="gallery-card__badge">${item.groups.includes("tables") ? "Таблица" : "Рисунок"}</span>
            <h3>${item.title}</h3>
            <p>${item.note}</p>
          </div>
        </a>
      `
    )
    .join("");

  observeReveal(galleryRoot);
}

function observeReveal(scope = document) {
  const items = Array.from(scope.querySelectorAll("[data-reveal]"));
  if (!items.length) {
    return;
  }

  if (!("IntersectionObserver" in window)) {
    items.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  items.forEach((item) => observer.observe(item));
}

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const nextFilter = button.dataset.filter ?? "all";
    filterButtons.forEach((candidate) => {
      candidate.classList.toggle("is-active", candidate === button);
      candidate.setAttribute("aria-pressed", String(candidate === button));
    });
    renderGallery(nextFilter);
  });
});

observeReveal();
renderGallery();
